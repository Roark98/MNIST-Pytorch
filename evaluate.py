import torch
import numpy as np
import torchvision
from models import *


train_transformations = transformations = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
        root='.', download=True, train=False, transform=train_transformations)

train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=False)


model = BranchCNN().to('cuda')
model.load_state_dict(torch.load('best_model_test.pth'))
model.eval()

total_hits = 0
for images, labels in train_dataloader:
    images = images.to('cuda')
    out = model(images)
    total_hits += torch.sum(out.cpu().argmax(dim=1)==labels.data)

print(total_hits/len(train_dataset))

