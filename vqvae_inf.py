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
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import random

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



    def __getitem__(self, idx):
        img = Image.open(self.data_list[idx])
        if self.transform:
            img = self.transform(img)
        label_index = self.cls_list.index(self.data_list[idx].split('/')[-2])

        
        return img, label_index
    
    def __len__(self):
        return len(self.data_list)

def select_table(feats, nums):
    size = int(np.sqrt(nums))
    code = torch.zeros(1,nums).long()
    for j in range(nums):
        a = random.random()
        print(a)
        idx = np.where(feats[j] >= a)[0]
        if len(idx) == 0:
            idx = 512
        else:
            idx = idx[0]
        code[0][j] = idx
    return code.view(1, size,size)

def save_img(sample, i):
    utils.save_image(
        sample,
        f"./class_{str(i).zfill(5)}.png",
        nrow=1,
        normalize=True,
        range=(-1, 1),
    )

def gen_table(feats, nums):
    size = int(np.sqrt(nums))
    code = torch.zeros(1,nums).long()
    for j in range(nums):
        idx = feats[j] * 512
        code[0][j] = int(idx)
    return code.view(1, size,size)


def inf(loader, model, device):
    model.eval()
    dis_t = torch.load('./class_features_t_10.pt')
    dis_b = torch.load('./class_features_b_10.pt')
    for i in range(10):
        feat_t = dis_t[i]
        feat_b = dis_b[i]
        code_t = gen_table(feat_t, 1024).cuda()
        code_b = gen_table(feat_b, 4096).cuda()
        out = model.decode_code(code_t, code_b)
        save_img(out, i)

    


    # if dist.is_primary():
        # loader = tqdm(loader)


    # list_id_t = [torch.zeros(1024) for _ in CLASSES]
    # list_id_b = [torch.zeros(4096) for _ in CLASSES]
    # num_per_cls = [0 for _ in CLASSES]
    # for i, (img, label) in enumerate(loader):

    #     img = img.to(device)

    #     out, id_t, id_b = model.inf(img)
    #     id_t = id_t.reshape(-1).cpu() / 512
    #     id_b = id_b.reshape(-1).cpu() / 512
    #     num_per_cls[label.item()] += 1
    #     list_id_t[label.item()] += id_t
    #     list_id_b[label.item()] += id_b
    # for i in range(len(num_per_cls)):
    #     list_id_t[i] /= num_per_cls[i]
    #     list_id_b[i] /= num_per_cls[i]
    #     list_id_t[i] = list_id_t[i].view(1, -1)
    #     list_id_b[i] = list_id_b[i].view(1, -1)


    # torch.save(torch.cat(list_id_t,0),'./class_features_t_10.pt')
    # torch.save(torch.cat(list_id_b,0), './class_features_b_10.pt')
    
        



    


        
        

        

        

            


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
        dataset, batch_size= 1, sampler=sampler, num_workers=2
    )

    model = VQVAE().to(device)


    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )
    pretrained = True
    if pretrained:
        weights = torch.load('./checkpoint/WVQVAE.pt')
        new_state_dict = OrderedDict()
        for k,v in weights.items():
            new_state_dict[k.replace('module.','')] = v 
        model.load_state_dict(new_state_dict)



    inf(loader, model, device)

        


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
