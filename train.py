import os
from model import CANNet2s
from utils import save_checkpoint

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

import numpy as np
import argparse
import json
import cv2
import dataset
import time

parser = argparse.ArgumentParser(description='PyTorch CANNet2s')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('val_json', metavar='VAL',
                    help='path to val json')

def main():
    global args,best_prec1

    best_prec1 = 1e6

    args = parser.parse_args()
    args.lr = 1e-4
    args.batch_size    = 1 
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 200
    args.workers = 4
    args.seed = int(time.time())
    args.print_freq = 1000
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.val_json, 'r') as outfile:
        val_list = json.load(outfile)

    torch.cuda.manual_seed(args.seed)

    model = CANNet2s()

    model = model.cuda()

    criterion = nn.MSELoss(size_average=False).cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.decay)

    for epoch in range(args.start_epoch, args.epochs):

        train(train_list, model, criterion, optimizer, epoch)
        prec1 = validate(val_list, model, criterion)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'state_dict': model.state_dict(),
        }, is_best)

def train(train_list, model, criterion, optimizer, epoch):

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),
                       train=True,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()

    for i,(prev_img, img, post_img, prev_target, target, post_target ) in enumerate(train_loader):

        data_time.update(time.time() - end)

        prev_img = prev_img.cuda()
        prev_img = Variable(prev_img)

        img = img.cuda()
        img = Variable(img)

        post_img = post_img.cuda()
        post_img = Variable(post_img)

        prev_flow = model(prev_img,img)
        post_flow = model(img,post_img)

        prev_flow_inverse = model(img,prev_img)
        post_flow_inverse = model(post_img,img)

        target = target.type(torch.FloatTensor)[0].cuda()
        target = Variable(target)

        prev_target = prev_target.type(torch.FloatTensor)[0].cuda()
        prev_target = Variable(prev_target)

        post_target = post_target.type(torch.FloatTensor)[0].cuda()
        post_target = Variable(post_target)

        # mask the boundary locations where people can move in/out between regions outside image plane
        mask_boundry = torch.zeros(prev_flow.shape[2:])
        mask_boundry[0,:] = 1.0
        mask_boundry[-1,:] = 1.0
        mask_boundry[:,0] = 1.0
        mask_boundry[:,-1] = 1.0

        mask_boundry = Variable(mask_boundry.cuda())


        reconstruction_from_prev = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry

        reconstruction_from_post = torch.sum(post_flow[0,:9,:,:],dim=0)+post_flow[0,9,:,:]*mask_boundry

        reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry

        reconstruction_from_post_inverse = F.pad(post_flow_inverse[0,0,1:,1:],(0,1,0,1))+F.pad(post_flow_inverse[0,1,1:,:],(0,0,0,1))+F.pad(post_flow_inverse[0,2,1:,:-1],(1,0,0,1))+F.pad(post_flow_inverse[0,3,:,1:],(0,1,0,0))+post_flow_inverse[0,4,:,:]+F.pad(post_flow_inverse[0,5,:,:-1],(1,0,0,0))+F.pad(post_flow_inverse[0,6,:-1,1:],(0,1,1,0))+F.pad(post_flow_inverse[0,7,:-1,:],(0,0,1,0))+F.pad(post_flow_inverse[0,8,:-1,:-1],(1,0,1,0))+post_flow_inverse[0,9,:,:]*mask_boundry


        prev_density_reconstruction = torch.sum(prev_flow[0,:9,:,:],dim=0)+prev_flow[0,9,:,:]*mask_boundry
        prev_density_reconstruction_inverse = F.pad(prev_flow_inverse[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow_inverse[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow_inverse[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow_inverse[0,3,:,1:],(0,1,0,0))+prev_flow_inverse[0,4,:,:]+F.pad(prev_flow_inverse[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow_inverse[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow_inverse[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow_inverse[0,8,:-1,:-1],(1,0,1,0))+prev_flow_inverse[0,9,:,:]*mask_boundry

        post_density_reconstruction_inverse = torch.sum(post_flow_inverse[0,:9,:,:],dim=0)+post_flow_inverse[0,9,:,:]*mask_boundry
        post_density_reconstruction = F.pad(post_flow[0,0,1:,1:],(0,1,0,1))+F.pad(post_flow[0,1,1:,:],(0,0,0,1))+F.pad(post_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(post_flow[0,3,:,1:],(0,1,0,0))+post_flow[0,4,:,:]+F.pad(post_flow[0,5,:,:-1],(1,0,0,0))+F.pad(post_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(post_flow[0,7,:-1,:],(0,0,1,0))+F.pad(post_flow[0,8,:-1,:-1],(1,0,1,0))+post_flow[0,9,:,:]*mask_boundry

        prev_reconstruction_from_prev = torch.sum(prev_flow[0,:9,:,:],dim=0)+prev_flow[0,9,:,:]*mask_boundry
        post_reconstruction_from_post = F.pad(post_flow[0,0,1:,1:],(0,1,0,1))+F.pad(post_flow[0,1,1:,:],(0,0,0,1))+F.pad(post_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(post_flow[0,3,:,1:],(0,1,0,0))+post_flow[0,4,:,:]+F.pad(post_flow[0,5,:,:-1],(1,0,0,0))+F.pad(post_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(post_flow[0,7,:-1,:],(0,0,1,0))+F.pad(post_flow[0,8,:-1,:-1],(1,0,1,0))+post_flow[0,9,:,:]*mask_boundry


        loss_prev_flow = criterion(reconstruction_from_prev, target)
        loss_post_flow = criterion(reconstruction_from_post, target)
        loss_prev_flow_inverse = criterion(reconstruction_from_prev_inverse, target)
        loss_post_flow_inverse = criterion(reconstruction_from_post_inverse, target)
        loss_prev = criterion(prev_reconstruction_from_prev,prev_target)
        loss_post = criterion(post_reconstruction_from_post,post_target)

        # cycle consistency
        loss_prev_consistency = criterion(prev_flow[0,0,1:,1:], prev_flow_inverse[0,8,:-1,:-1])+criterion(prev_flow[0,1,1:,:], prev_flow_inverse[0,7,:-1,:])+criterion(prev_flow[0,2,1:,:-1], prev_flow_inverse[0,6,:-1,1:])+criterion(prev_flow[0,3,:,1:], prev_flow_inverse[0,5,:,:-1])+criterion(prev_flow[0,4,:,:], prev_flow_inverse[0,4,:,:])+criterion(prev_flow[0,5,:,:-1], prev_flow_inverse[0,3,:,1:])+criterion(prev_flow[0,6,:-1,1:], prev_flow_inverse[0,2,1:,:-1])+criterion(prev_flow[0,7,:-1,:], prev_flow_inverse[0,1,1:,:])+criterion(prev_flow[0,8,:-1,:-1], prev_flow_inverse[0,0,1:,1:])

        loss_post_consistency = criterion(post_flow[0,0,1:,1:], post_flow_inverse[0,8,:-1,:-1])+criterion(post_flow[0,1,1:,:], post_flow_inverse[0,7,:-1,:])+criterion(post_flow[0,2,1:,:-1], post_flow_inverse[0,6,:-1,1:])+criterion(post_flow[0,3,:,1:], post_flow_inverse[0,5,:,:-1])+criterion(post_flow[0,4,:,:], post_flow_inverse[0,4,:,:])+criterion(post_flow[0,5,:,:-1], post_flow_inverse[0,3,:,1:])+criterion(post_flow[0,6,:-1,1:], post_flow_inverse[0,2,1:,:-1])+criterion(post_flow[0,7,:-1,:], post_flow_inverse[0,1,1:,:])+criterion(post_flow[0,8,:-1,:-1], post_flow_inverse[0,0,1:,1:])


        loss = loss_prev_flow+loss_post_flow+loss_prev_flow_inverse+loss_post_flow_inverse+loss_prev+loss_post+loss_prev_consistency+loss_post_consistency

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def validate(val_list, model, criterion):
    print ('begin val')
    val_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=1)

    model.eval()

    mae = 0

    for i,(prev_img, img, post_img, _, target,_ ) in enumerate(val_loader):
        # only use previous frame in inference time, as in real-time application scenario, future frame is not available
        prev_img = prev_img.cuda()
        prev_img = Variable(prev_img)

        img = img.cuda()
        img = Variable(img)

        prev_flow = model(prev_img,img)
        prev_flow_inverse = model(img,prev_img)


        target = target.type(torch.FloatTensor)[0].cuda()
        target = Variable(target)

        mask_boundry = torch.zeros(prev_flow.shape[2:])
        mask_boundry[0,:] = 1.0
        mask_boundry[-1,:] = 1.0
        mask_boundry[:,0] = 1.0
        mask_boundry[:,-1] = 1.0

        mask_boundry = Variable(mask_boundry.cuda())


        reconstruction_from_prev = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry

        reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry

        overall = ((reconstruction_from_prev+reconstruction_from_prev_inverse)/2.0).type(torch.FloatTensor)

        target = target.type(torch.FloatTensor)

        mae += abs(overall.data.sum()-target.sum())

    mae = mae/len(val_loader)
    print(' * MAE {mae:.3f} '
              .format(mae=mae))

    return mae

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
