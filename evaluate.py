import torch
import numpy as np
import torchvision
from models import *
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='CNN', help='Model to use',
                        choices={'MLP', 'CNN', 'BranchCNNLarge', 'BranchCNNShort'})

    args = parser.parse_args()

    transformations = transformations = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root='.', download=True, train=True, transform=transformations)

    test_dataset = torchvision.datasets.MNIST(
        root='.', download=True, train=False, transform=transformations)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=False)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False)

    datasets = {'train': train_dataset, 'test': test_dataset}
    dataloaders = {'train': train_dataloader, 'test': test_dataloader}

    if args.model == 'MLP':
        model = MLP().to('cuda')
    elif args.model == 'CNN':
        model = CNN().to('cuda')
    elif args.model == 'BranchCNNLarge':
        model = BranchCNNLarge().to('cuda')
    else:
        model = BranchCNNShort().to('cuda')

    model_path = os.path.join("results", args.model, 'best_model_test.pth')


    model.load_state_dict(torch.load(model_path))
    model.eval()

    results = {}

    for mode in ["train", "test"]:
        total_hits = 0
        for images, labels in dataloaders[mode]:
            images = images.to('cuda')
            out = model(images)
            total_hits += torch.sum(out.cpu().argmax(dim=1) == labels.data)
        results[mode] = total_hits/len(datasets[mode])

    print('Train accuracy: ', round(results['train'].item(), 4))
    print('Test accuracy: ', round(results['test'].item(), 4))
