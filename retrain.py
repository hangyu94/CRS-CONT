from __future__ import print_function

import sys
import argparse
import time
import math
import os

import torch
import torch.backends.cudnn as cudnn
from torch.utils import data
from torchvision import transforms, datasets
from torch.optim import lr_scheduler

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, LinearClassifier
from dataset_loader import Dataset_RAF, Dataset_AffectNet, FERPlus
from sampler import ImbalancedDatasetSampler

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=4, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        choices=['0.2'], help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='SFEW',
                        choices=['RAF-DB', 'AffectNet', 'SFEW', 'FERPlus', 'CK', 'FED-RO'], help='dataset')
    # RAF-DB setting
    parser.add_argument('--RAF-train-root', type=str, default='/home/gpu/FER/datasets/RAFdataset/train',
                        help="root path to train data directory")
    parser.add_argument('--RAF-test-root', type=str, default='/home/gpu/FER/datasets/RAFdataset/test',
                        help="root path to test data directory")
    parser.add_argument('--RAF-label-train-txt', default='/home/gpu/FER/datasets/RAFdataset/RAF_train_label2.txt', type=str, help='')
    parser.add_argument('--RAF-label-test-txt', default='/home/gpu/FER/datasets/RAFdataset/RAF_test_label2.txt', type=str, help='')
    # AffectNet setting
    parser.add_argument('--Aff-root', type=str, default='/home/gpu/FER/datasets/Manually_Annotated_Images_Crop1308',
                        help="root path to train data directory")
    parser.add_argument('--Aff-label-train-txt', default='/home/gpu/FER/datasets/training_label2.csv', type=str, help='')
    parser.add_argument('--Aff-label-test-txt', default='/home/gpu/FER/datasets/validation_label2.csv', type=str, help='')
    # SFEW setting
    parser.add_argument('--SFEW-train-root', type=str, default='/home/gpu/FER/datasets',
                        help="root path to train data directory")
    parser.add_argument('--SFEW-test-root', type=str, default='/home/gpu/FER/datasets/SFEW/Val_crop',
                        help="root path to test data directory")
    parser.add_argument('--SFEW-label-train-txt', default='/home/gpu/FER/datasets/SFEW/train.txt', type=str, help='')
    parser.add_argument('--SFEW-label-test-txt', default='/home/gpu/FER/datasets/SFEW/val.txt', type=str, help='')
    # FERPlus setting
    parser.add_argument('--file-name', type=str, default='/home/gpu/FER/datasets/FER++/FERPlus1.h5', help="root path to train data directory")
    # FED-RO setting
    parser.add_argument('--FED_test_root', default='/home/gpu/FER/datasets/FED-RO_crop1', type=str, help='')
    parser.add_argument('--FED-label-test-txt', default='/home/gpu/FER/datasets/FED-RO_crop/FED-RO.txt', type=str, help='')
    # CK+ setting
    parser.add_argument('--CK-test-root', type=str, default='/home/gpu/FER/datasets/CK+_crop',
                        help="root path to test data directory")
    parser.add_argument('--CK-label-test-txt', default='/home/gpu/FER/datasets/CK+_crop/CK+_8.txt', type=str,
                        help='')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='save/Mix_models/CRS-CONT_Mix_resnet18_bsz_128/last.pth',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
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

    if opt.dataset == 'RAF-DB':
        opt.n_cls = 7
    elif opt.dataset == 'AffectNet':
        opt.n_cls = 7
    elif opt.dataset == 'SFEW':
        opt.n_cls = 7
    elif opt.dataset == 'CK':
        opt.n_cls = 8
    elif opt.dataset == 'FERPlus':
        opt.n_cls = 8
    elif opt.dataset == 'FED-RO':
        opt.n_cls = 7
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt

def set_loader(opt):
    # construct data loader
    if opt.dataset == 'RAF-DB' or 'AffectNet' or 'SFEW' or 'CAER-S' or 'CK' or 'FERPlus':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.6, 1.)),#0.2 1
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'RAF-DB':
        train_dataset = Dataset_RAF(root=opt.RAF_train_root, file_list=opt.RAF_label_train_txt, transform=train_transform)
        val_dataset = Dataset_RAF(root=opt.RAF_test_root, file_list=opt.RAF_label_test_txt, transform=val_transform)
    elif opt.dataset == 'AffectNet':
        train_dataset = Dataset_AffectNet(root=opt.Aff_root, file_list=opt.Aff_label_train_txt, transform=train_transform)
        val_dataset = Dataset_AffectNet(root=opt.Aff_root, file_list=opt.Aff_label_test_txt, transform=val_transform)
    elif opt.dataset == 'SFEW':
        train_dataset = Dataset_RAF(root=opt.SFEW_train_root, file_list=opt.SFEW_label_train_txt, transform=train_transform)
        val_dataset = Dataset_RAF(root=opt.SFEW_test_root, file_list=opt.SFEW_label_test_txt, transform=val_transform)
        # train_dataset_RAF = Dataset_RAF(root=opt.RAF_train_root, file_list=opt.RAF_label_train_txt, transform=train_transform)
        # train_dataset_Aff = Dataset_AffectNet(root=opt.Aff_root, file_list=opt.Aff_label_train_txt, transform=train_transform)
        # train_dataset = data.ConcatDataset([train_dataset_RAF, train_dataset_Aff])
        # val_dataset1 = Dataset_RAF(root=opt.SFEW_train_root, file_list=opt.SFEW_label_train_txt, transform=train_transform)
        # val_dataset2 = Dataset_RAF(root=opt.SFEW_test_root, file_list=opt.SFEW_label_test_txt, transform=val_transform)
        # val_dataset = data.ConcatDataset([val_dataset1, val_dataset2])
    elif opt.dataset == 'FED-RO':
        train_dataset_RAF = Dataset_RAF(root=opt.RAF_train_root, file_list=opt.RAF_label_train_txt, transform=train_transform)
        train_dataset_Aff = Dataset_AffectNet(root=opt.Aff_root, file_list=opt.Aff_label_train_txt, transform=train_transform)
        train_dataset = data.ConcatDataset([train_dataset_RAF, train_dataset_Aff])
        val_dataset = Dataset_RAF(root=opt.FED_test_root, file_list=opt.FED_label_test_txt, transform=val_transform)
    elif opt.dataset == 'CK':
        # train_dataset_RAF = Dataset_RAF(root=opt.RAF_train_root, file_list=opt.RAF_label_train_txt, transform=train_transform)
        # train_dataset_Aff = Dataset_AffectNet(root=opt.Aff_root, file_list=opt.Aff_label_train_txt, transform=train_transform)
        # train_dataset = data.ConcatDataset([train_dataset_RAF, train_dataset_Aff])
        #train_dataset = Dataset_RAF(root=opt.RAF_train_root, file_list=opt.RAF_label_train_txt, transform=train_transform)
        train_dataset = Dataset_AffectNet(root=opt.Aff_root, file_list=opt.Aff_label_train_txt, transform=train_transform)
        val_dataset = Dataset_RAF(root=opt.CK_test_root, file_list=opt.CK_label_test_txt, transform=val_transform)
    elif opt.dataset == 'FERPlus':
        train_dataset = FERPlus(file_name=opt.file_name, split='Training', transform=train_transform)
        val_dataset = FERPlus(file_name=opt.file_name, split='PrivateTest', transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    if opt.dataset == 'RAF-DB':
        train_sampler = None
    elif opt.dataset == 'AffectNet':
        train_sampler = ImbalancedDatasetSampler(train_dataset)
    elif opt.dataset == 'SFEW':
        train_sampler = None
    elif opt.dataset == 'FED-RO':
        train_sampler = None
    elif opt.dataset == 'CK':
        train_sampler = None
    elif opt.dataset == 'FERPlus':
        train_sampler = None
    else:
        raise ValueError(opt.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader

def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)

        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels, _) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.encoder.parameters())/1000000.0))

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(classifier, optimizer, opt, epoch, os.path.join('best_classifier.pth'))

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
