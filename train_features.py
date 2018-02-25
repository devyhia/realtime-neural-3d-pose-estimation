import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import random

from models.features import Features
from models.triplets_loss import TripletsLoss
from datasets.training import TrainingDataset
from helpers.logger import setup_logger

# Training settings
parser = argparse.ArgumentParser(description='Feature Extractor Trainer')
parser.add_argument('--batch-size', type=int, default=16, help='input batch size for training (default: 64)')
parser.add_argument('--dataset', default='/Users/yehyaa/Downloads/dataset/', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.01)')
parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
parser.add_argument('--num-workers', type=int, default=2, help='how many workers for data loading')
parser.add_argument('--manual-seed', type=int, default=800, help='manual seed for random number generators')

use_gpu = torch.cuda.is_available()

if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    random.seed(args.manual_seed)

    if use_gpu:
        torch.cuda.manual_seed(args.manual_seed)

    # Set up logger
    logger = setup_logger()

    logger.info(args)

    # Load Dataset & Batch Loader
    logger.info("Loading the dataset ...")
    dataset = TrainingDataset(args.dataset)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Load Model
    logger.info("Loading the model ...")
    model = Features()

    # Triplets Loss
    logger.info("Loading the triplets loss function ...")
    triplets_loss = TripletsLoss()

    # Enable GPU
    if use_gpu:
        model = model.cuda()
        triplets_loss = triplets_loss.cuda()

    # Set model in training mode
    logger.info("Setting up training mode ...")
    model.train()

    # Adam Optimizer
    logger.info("Creating Adam Optimizer ...")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Start Training
    for epoch in range(1, args.epochs + 1):
        for batch_idx, data in enumerate(train_loader):
            anchors, pullers, pushers = \
                data['anchor'], data['puller'], data['pusher']

            if use_gpu:
                anchors, pullers, pushers = \
                    Variable(anchors.cuda()), Variable(pullers.cuda()), Variable(pushers.cuda())
            else:
                anchors, pullers, pushers = \
                    Variable(anchors), Variable(pullers), Variable(pushers)

            optimizer.zero_grad()
            features = model(anchors, pullers, pushers)

            loss = triplets_loss(*features)
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % args.log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]
                ))
        
        # Save Model After Each Epoch
        torch.save(model.state_dict(), 'model.epoch.{}.pth'.format(epoch))