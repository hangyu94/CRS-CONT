from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.optim import lr_scheduler

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss
from dataset_loader import Dataset_RAF, Dataset_AffectNet

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=500,
                        help='print frequency')#20
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.5, choices=[1.0, 0.5],
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--stepsize', default=20, type=int, choices=['20'],
                        help="stepsize to decay learning rate (>0 means this is enabled)")
    parser.add_argument('--gamma', default=0.5, type=float,
                        help="learning rate decay")

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='Mix',
                        choices=['RAF-DB', 'AffectNet'], help='dataset')

    parser.add_argument('--RAF-train-root', type=str, default='/home/gpu/FER/datasets/RAFdataset/train',
                        help="root path to train data directory")
    parser.add_argument('--RAF-label-train-txt', default='/home/gpu/FER/datasets/RAFdataset/RAF_train_label2.txt', type=str, help='')
    parser.add_argument('--Aff-train-root', type=str, default='/home/gpu/FER/datasets/Manually_Annotated_Images_Crop130',
                        help="root path to train data directory")
    parser.add_argument('--Aff-label-train-txt', default='/home/gpu/FER/datasets/training_label1.csv', type=str, help='')

    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='CRS-CONT',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07, choices=['0.07'],
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()


    # set the path according to the environment
    opt.model_path = './save/{}_models'.format(opt.dataset)
    opt.tb_path = './save/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_bsz_{}'.format(opt.method, opt.dataset, opt.model, opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size >= 512:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'RAF-DB' or 'AffectNet' or 'Mix':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.5, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'RAF-DB':
        train_dataset = Dataset_RAF(root=opt.RAF_train_root, file_list=opt.RAF_label_train_txt, transform=TwoCropTransform(train_transform))
    elif opt.dataset == 'AffectNet':
        train_dataset = Dataset_AffectNet(root=opt.Aff_train_root, file_list=opt.Aff_label_train_txt, transform=TwoCropTransform(train_transform))
    elif opt.dataset == 'Mix':
        train_dataset_RAF = Dataset_RAF(root=opt.RAF_train_root, file_list=opt.RAF_label_train_txt, transform=TwoCropTransform(train_transform))
        train_dataset_Aff = Dataset_AffectNet(root=opt.Aff_train_root, file_list=opt.Aff_label_train_txt, transform=TwoCropTransform(train_transform))
        train_dataset = data.ConcatDataset([train_dataset_RAF, train_dataset_Aff])
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)
    criterion_weak = nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        criterion_weak = criterion_weak.cuda()
        cudnn.benchmark = True

    return model, criterion, criterion_weak


def train(train_loader, model, criterion, criterion_weak, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels, coarse_labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            coarse_labels = coarse_labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        features, coarse_output = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        c1, c2 = torch.split(coarse_output, [bsz, bsz], dim=0)
        c = torch.cat([c1.unsqueeze(1), c2.unsqueeze(1)], dim=1)

        if opt.method == 'SupCon':
            loss1 = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss1 = criterion(features)
        elif opt.method == 'CRS-CONT':
            loss1 = criterion(features, coarse_labels)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))
        c = F.softmax(c, dim=2)
        loss = loss1 + criterion_weak(c1, coarse_labels) + criterion(F.normalize(c, dim=2))*0.4

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion, criterion_weak = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    if opt.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        #adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, criterion_weak, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        scheduler.step()

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
