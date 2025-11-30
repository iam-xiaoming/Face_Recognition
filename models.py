import torch
from mobilefacenet import get_mbf
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import os
import shutil


def load_model(device_name='windows'):
    global device
    if device_name == 'windows':
        device = torch.device('cuda' if torch.mps.is_available() else "cpu")
    else:
        device = torch.device('mps' if torch.mps.is_available() else "cpu")
    
    model = get_mbf(fp16=False, num_features=512).to(device)

    checkpoint = torch.load("models/mbf_backbone.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def get_embedding(model, img, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x)
        emb = F.normalize(emb, p=2, dim=1)

    return emb.cpu().numpy()[0]


def cosine(a, b):
    return float(np.dot(a, b))


def _l2normalize(v):
    return v / (np.linalg.norm(v) + 1e-6)

def average_embeddings(v):
    v_mean = np.mean(v, axis=1)
    return _l2normalize(v_mean)


DIR = 'embeddings'
def save(username, v):
    filename = os.path.join(DIR, username)
    if os.path.exists(filename):
        os.remove(filename)
    np.save(filename, v)
    
def load(path):
    v = np.load(path)
    return v

def loads():
    embs = []
    users = []
    for file in os.listdir(DIR)[:10]:
        path = os.path.join(DIR, file)
        v = load(path)
        embs.append(v)
        users.append(file.replace('.npy', ''))
    return embs, users