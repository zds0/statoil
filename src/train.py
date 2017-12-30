"""Perform training."""

import os
import time

import click
import torch
from numpy import mean
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import Adam
from tqdm import tqdm

from data import prep_data
from nets import Net, Net2
from utils import MetricMeter, accuracy

# settings
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 32
THREADS = 4 # for the DataLoaders
USE_CUDA = True if torch.cuda.is_available() else False

# training
def train_model(epochs, net, criterion, optimizer, early_stop):
    """Perform the training."""
    # DataLoaders
    train_loader, val_loader = prep_data(BATCH_SIZE, THREADS, USE_CUDA)

    net.train()
    loss_list = []

    for epoch in tqdm(range(epochs), total=epochs):
        running_loss = MetricMeter()
        running_accuracy = MetricMeter()
        val_loss_meter = MetricMeter()
        val_acc_meter = MetricMeter()

        for i, dict_ in enumerate(train_loader):
            images = dict_['img']
            target = dict_['target'].type(torch.FloatTensor)#.long()

            if USE_CUDA:
                images = images.cuda()
                target = target.cuda()

            images = Variable(images)
            target = Variable(target)

            output = net(images).squeeze()
            loss = criterion(output, target)
            acc = accuracy(target.data, output.data)
            running_loss.update(loss.data[0])
            running_accuracy.update(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for i, dict_ in enumerate(val_loader):
            images = dict_['img']
            target = dict_['target'].type(torch.FloatTensor)#.long()

            if USE_CUDA:
                images = images.cuda()
                target = target.cuda()

            images = Variable(images)
            target = Variable(target)

            output = net(images).squeeze()
            val_loss = criterion(output, target)
            val_acc = accuracy(target.data, output.data)
            val_loss_meter.update(val_loss.data[0])
            val_acc_meter.update(val_acc)

        # hacky early stopping...
        loss_list.append(val_loss_meter.avg)
        if (mean(loss_list[-12:-4]) < mean(loss_list[-4:])) and (len(loss_list) > 12) and early_stop:
            print('Validation loss no longer decreasing... stopping training.')
            break

        print("[ loss: {:.4f} | acc: {:.4f} | vloss: {:.4f} | vacc: {:.4f} ]".format(
            running_loss.avg, running_accuracy.avg, val_loss_meter.avg, val_acc_meter.avg))

    ts = time.strftime("%Y-%m-%d_%H-%M")
    vl = '_{:.5f}'.format(val_loss_meter.avg)
    fname = 'model_' + ts + vl + '.pth'
    torch.save(net, os.path.join(THIS_DIR, '..', 'weights', fname))


@click.command()
@click.option('--epochs', type=int, default=50)
@click.option('--early_stop', type=bool, default=False)
def main(epochs, early_stop):
    # net = Net()
    net = Net2()

    if USE_CUDA:
        net.cuda()

    # if not USE_CUDA:
    #     epochs = 1 # for testing on laptop

    # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
    criterion = BCELoss()

    # adding weight_decay is a form of L2 regularization.
    # See: https://discuss.pytorch.org/t/simple-l2-regularization/139
    optimizer = Adam(net.parameters(), weight_decay=1e-5)#, lr=1e-3)

    train_model(epochs, net, criterion, optimizer, early_stop)


if __name__ == '__main__':
    main()
