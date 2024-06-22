#!/usr/bin/env python
from numpy import genfromtxt
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])

data = datasets.ImageFolder('/Users/yourusername/storage/pathtrack_release/train/', transform=transform)

dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False)

dataiter = iter(dataloader)
images, _ = dataiter.next()

print(data)

boxes = genfromtxt('/Users/yourusername/storage/pathtrack_release/train/-DGzHCfmv5k_23_30/gt/gt.txt')
