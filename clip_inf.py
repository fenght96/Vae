import os
import clip
import pdb
import torch

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, preprocess = clip.load('ViT-B/32', device)

text_inputs = torch.cat([clip.tokenize(f'a photo of a {c}') for c in classes]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    print(type(text_features))
    torch.save(text_features.cpu(), './Classes_features_10.pt')
    print(f'text features:{text_features.shape}')
    # pdb.set_trace()