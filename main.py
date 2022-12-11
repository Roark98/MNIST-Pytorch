import torchvision
import torch
import argparse
# import warnings
from torchsummary import summary
from models import *
from train_routine import train
import time

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
                        default='CNN', help='Model to use', choices={'MLP', 'CNN', 'BranchCNNLarge','BranchCNNShort'})

    args = parser.parse_args()

    transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.RandomRotation(45)])
    test_transformations = transformations = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root='.', download=True, train=True, transform=transformations)
    test_dataset = torchvision.datasets.MNIST(
        root='.', download=True, train=False, transform=test_transformations)
    datasets = {'train': train_dataset, 'test': test_dataset}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=12, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=12, shuffle=False)


    if args.model=='MLP':
        model = MLP().to('cuda')
    elif args.model=='CNN':
        model = CNN().to('cuda')
    elif args.model=='BranchCNNLarge':
        model = BranchCNNLarge().to('cuda')
    else:
        model = BranchCNNShort().to('cuda')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    start = time.time()

    train(32, model, args.model, criterion, optimizer, train_dataloader, test_dataloader, datasets)
    print("Total training time: " +str(time.time()-start))
