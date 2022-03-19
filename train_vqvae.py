import argparse
import sys
import os
import glob
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from torchvision import datasets, transforms, utils

from tqdm import tqdm
import pdb
from vqvae import VQVAE, WVQVAE
from scheduler import CycleScheduler
import distributed as dist
from PIL import Image
from collections import OrderedDict

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow')

class ReadData(Dataset):
    def __init__(self, root='./', transform=None):
        super(ReadData, self).__init__()
        data_list = glob.iglob(os.path.join(root, "*/*.jpg"))
        self.data_list = []
        self.cls_list = CLASSES#os.listdir(root)
        for x in data_list:
            if str(x).split('/')[-2] in self.cls_list:
                self.data_list.append(x)
        self.transform = transform
        self.cls_features = torch.load('./Classes_features_10.pt')


    def __getitem__(self, idx):
        img = Image.open(self.data_list[idx])
        if self.transform:
            img = self.transform(img)
        label_index = self.cls_list.index(self.data_list[idx].split('/')[-2])
        label_feature = self.cls_features[label_index]
        var = torch.zeros(label_feature.shape, dtype=torch.float64)
        var = var + (0.1**0.5)*torch.randn(label_feature.shape)
        label_feature += var
        return img, label_feature
    
    def __len__(self):
        return len(self.data_list)


def train(epoch, loader, model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)
        label = label.to(device)

        out, latent_loss = model(img, label)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            if i % 100 == 0:
                model.eval()

                sample = label[:sample_size]
                sample_img = img[:sample_size]

                with torch.no_grad():
                    out, _ = model(sample_img, sample)
                    out_sample = model.forward_label(sample)

                utils.save_image(
                    torch.cat([sample_img, out], 0),
                    f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )
                utils.save_image(
                    torch.cat([sample_img, out_sample], 0),
                    f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_label.png",
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )

                model.train()


def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1



    dataset = ReadData('/home/fht/data/VOCdevkit/VOC2007/classification/',
            transform = transforms.Compose([
                        transforms.Resize((args.size, args.size)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=128 // args.n_gpu, sampler=sampler, num_workers=2
    )

    model = WVQVAE().to(device)


    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )
    pretrained = False
    if pretrained:
        weights = torch.load('./checkpoint/WVQVAE.pt')
        new_state_dict = OrderedDict()
        for k,v in weights.items():
            new_state_dict[k.replace('module.','')] = v 
        model.load_state_dict(new_state_dict)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)

        if dist.is_primary():
            torch.save(model.state_dict(), f"./checkpoint/WVQVAE.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
