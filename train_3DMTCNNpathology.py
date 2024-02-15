from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import sys
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, datasets 
from model_survival_pathology import MTCNN3Dpathology
import numpy as np
from PIL import Image
from utils import load_nifti_img
from utils import calculate_concordance_index

beta_attention = 0#0.1 
IMG_EXTENSIONS = ('nii.gz', 'nii', '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
foldnumber = '1'
weight_c = 1
weight_m = 1 

foldnumbers = ['1', '2', '3', '4', '5']
weight_ms = [1]
def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L    

def nifti_loader(path):
    #print(path)
    cube = load_nifti_img(path, dtype=np.float32)
    #print('----------cube value---------')
    #print(cube.max())
    #print(cube.min())
    #print(path)

    #print('shape of the cube')
    #print(np.shape(cube))
    return np.expand_dims(cube, axis=0)

def normalize(cube):
    #Cube window thresholding with Window width, window level = 400, 50
    low = -350 
    high = 450
    low_indices = cube < low
    high_indices = cube > high
    cube[low_indices] = low
    cube[high_indices] = high
    cube = (cube - low)/(high - low)
                    
    return cube

def parsepid(path_cc):
    parts = path_cc.split('/')
    pid =  parts[-1][:-7]
    pid = pid.lstrip('0')
    return pid

def parsepid_withzerosahead(path_cc):
    parts = path_cc.split('/')
    pid =  parts[-1][:-7]
    return pid
 

def load_pid2label():
    dictionary = json.load(open('pid2OS.json', 'r'))
    return dictionary

def load_pid2pTN():
    dictionary = json.load(open('data/jsons/pid2pTN.json', 'r'))
    return dictionary


class ImageFolderTwin(datasets.DatasetFolder):
    '''
    We overide the DatasetFolder class, instead of return (image, label) we return (image_data, image_roi)
    For each imaging data, find the corresponding mask and return them in a tuple (image_data, image_roi)
    '''
   
    def __getitem__(self, index):
        path_data, target = self.samples[index]
        image_data = self.loader(path_data)
        image_data = normalize(image_data)
        pid_withzerosahead = parsepid_withzerosahead(path_data)
        pid = parsepid(path_data)

        path_roi = 'data/twin/roi/' + pid_withzerosahead + '.nii.gz' 
        image_roi = self.loader(path_roi)
        image_roi = (image_roi > 0.5).astype(np.float32)
        image_data = torch.tensor(image_data)
        image_roi = torch.tensor(image_roi)
        labels = self.pid2label[pid]
        pTNs = self.pid2pTN[pid]
        return (image_data, image_roi, labels, pTNs)
    def __init__(self, root,batch_size=8, transform=None, target_transform=None,
        loader=nifti_loader, is_valid_file=None):
        super(ImageFolderTwin, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
        transform=transform,
        target_transform=target_transform,
        is_valid_file=is_valid_file)
        self.pid2label = load_pid2label()
        self.pid2pTN = load_pid2pTN()


def dice_loss(pred, target, smooth = 1.):
    #'https://github.com/usuyama/pytorch-unet/blob/master/loss.py'
    target_size = target.size() 
    #Upsample feature map from [100, 1, 16, 16] to [100, 1, 64, 64] 
    pred = F.upsample(pred, size = target_size[2:], mode='trilinear')
    pred = pred.contiguous()
    pred = torch.sigmoid(pred)
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()
    

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


def save_checkpoint(state, is_best, epoch, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, str(epoch) + filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, str(epoch) + filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = MVAE(checkpoint['n_latents'])
    model.load_state_dict(checkpoint['state_dict'])
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-latents', type=int, default=128,
                        help='size of the latent embedding [default: 128]')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training [default: 100]')#16
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train [default: 50]')
    parser.add_argument('--annealing-epochs', type=int, default=10, metavar='N',#1 is bug free
                        help='number of epochs to anneal KL for [default: 10]')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate [default: 1e-4]')#1e-4 is best for other folders
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--lambda-image', type=float, default=1,
                        help='multipler for image reconstruction [default: 1]')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training [default: False]')
    args = parser.parse_args()
    args.cuda = True# args.cuda and torch.cuda.is_available()

    if not os.path.isdir('./trained_models'):
        os.makedirs('./trained_models')

    preprocess_data = transforms.Compose([transforms.ToTensor()])

    print('preparing for dataloader')
    data_dir = 'data/'
    image_datasets = {x: ImageFolderTwin(os.path.join(data_dir, x), batch_size=args.batch_size, transform=preprocess_data) for x in ['train' + foldnumber, 'test' + foldnumber, 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=1) for x in ['train' + foldnumber,'test' + foldnumber, 'val']}
    train_loader = dataloaders['train' + foldnumber]
    test_loader = dataloaders['test' + foldnumber]
    val_loader = dataloaders['val']
    N_mini_batches = len(train_loader)

    def train(epoch, current_annealing_factor):

        model.train()
        train_loss_meter = AverageMeter()
        total_mse_meter = AverageMeter() 
        total_dice_meter = AverageMeter() 

        for batch_idx, (image, roi, labels, pTN)  in enumerate(train_loader):
            annealing_factor = current_annealing_factor

            if args.cuda:
                image     = image.cuda()
                roi = roi.cuda()
            image      = Variable(image)
            roi = Variable(roi)
            batch_size = len(image)

            # refresh the optimizer
            optimizer.zero_grad()

            # pass data through model
            preds, attention  = model(image)
            d_loss = dice_loss(attention, roi)

            # compute binary-cross entropy loss for each task 
            c_loss = torch.tensor(0.0).to(device) 
            num_tasks = len(labels) + 1 
            mse_loss = torch.tensor(0.0).to(device) 


            pT = pTN[0].cuda()
            pT = pT.to(device)
            for j in range(num_tasks):
                outputs = preds[j]
                if j < 6:
                    #print(labels[j])
                    label_task = labels[j].cuda()
                    label_task = label_task.to(device)
                    loss_task = criterion(outputs, label_task)
                    c_loss += loss_task 
                else:
                    '''
                    print('predicted pT = ')
                    print(outputs)
                    print('ground truth pT = ')
                    print(pT)
                    print('-' * 20)
                    '''
                    mse_loss = criterion_mse(outputs, pT.float())#comparing with ground truth pT 

            train_loss = weight_c * c_loss + weight_m * mse_loss 
             
            train_loss_meter.update(train_loss.item(), batch_size)
            total_mse_meter.update(mse_loss.item(), batch_size) 
            total_dice_meter.update(d_loss.item(), batch_size) 
            '''
            if batch_idx % 100 == 0:
                print('  c_loss = ' + str(c_loss.item()))
                print('  mse_loss = ' + str(mse_loss.item()))
                #print('train_loss = ' + str(train_loss))
                print('  dice loss = ' + str(d_loss.item()))
            '''

            
            # compute and take gradient step
            train_loss.backward()
            optimizer.step()

            '''
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(image), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), train_loss_meter.avg))
            '''

        '''
        print('====> Epoch: {}\t total Loss: {:.4f}'.format(epoch, train_loss_meter.avg))
        print('====> Epoch: {}\t MSE Loss: {:.4f}'.format(epoch, total_mse_meter.avg))
        '''


    def test(epoch):
        model.eval()

        test_loss_meter = AverageMeter()
        test_mse_meter = AverageMeter() 
        test_dice_meter = AverageMeter() 
        fnames = os.listdir('data/test' + foldnumber + '/raw/')
        num_test = len(fnames) 

        num_tasks = 6
        predictions = np.zeros(shape = (num_test, num_tasks))
        groundtruth = np.zeros(shape = (num_test, num_tasks))


        pbar = tqdm(total=num_test)
        for batch_idx, (image, roi, labels, pTN) in enumerate(test_loader):
            if args.cuda:
                image     = image.cuda()
                roi = roi.cuda()
            image      = Variable(image)
            roi = Variable(roi)
            pT = pTN[0].cuda()
            pT = pT.to(device)


            batch_size = len(image)
            # pass data through model
            preds, attention = model(image)
            d_loss = dice_loss(attention, roi)
            # encourage the attention to be focused on the tumor region
            # compute binary-cross entropy loss for each task 
            c_loss = torch.tensor(0.0).to(device) 
            mse_loss = torch.tensor(0.0).to(device) 
            num_tasks = len(labels) + 1

            for j in range(num_tasks):
                outputs = preds[j]
                if j < len(labels): 
                    label_task = labels[j].to(device)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    start_index = batch_idx * batch_size 
                    index_increase = len(predicted.data.tolist())
                    predictions[start_index : start_index + index_increase,j] = predicted.cpu()
                    groundtruth[start_index : start_index + index_increase,j] = label_task.cpu()

                    loss_task = criterion(outputs, label_task)
                    c_loss += loss_task 
                else:
                    mse_loss = criterion_mse(outputs, pT.float())#comparing with ground truth pT 



            test_loss = weight_c * c_loss + weight_m * mse_loss 

            test_loss_meter.update(test_loss.item(), batch_size)
            test_mse_meter.update(mse_loss.item(), batch_size)
            test_dice_meter.update(d_loss.item(), batch_size)
            pbar.update()

        pbar.close()

        c_index = calculate_concordance_index(predictions.tolist(), groundtruth.tolist())
        print('====> Test Loss: {:.4f}, MSE: {:.4f}, C-Index: {:.4f}'.format(test_loss_meter.avg, test_mse_meter.avg, c_index))
        return test_loss_meter.avg, c_index, test_mse_meter.avg, test_dice_meter.avg

    
    print('start training and testing')
    for foldnumber in foldnumbers:
        for weight_m in weight_ms:
            print('preparing model')
            model     = MTCNN3Dpathology(args.n_latents)
            print('model is prepared')
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            if args.cuda:
                model.cuda()

            best_loss = float('inf') 
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
            criterion_mse = nn.MSELoss()

            beta_np_cyc = frange_cycle_linear(0.0, 1.0, args.epochs + 1, 10)

            highest_cindex = 0.0
            corresponding_mse = 0.0
            corresponding_dice = 0.0
            for epoch in range(1, args.epochs + 1):
                current_annealing_factor = beta_np_cyc[epoch]
                train(epoch, current_annealing_factor)
                loss, cindex, mse, d_l      = test(epoch)
                if cindex > highest_cindex:
                    highest_cindex = cindex
                    corresponding_mse = mse
                    corresponding_dice = 1 - d_l
                is_best   = loss < best_loss
                best_loss = min(loss, best_loss)
                # save the best model and current model
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'n_latents': args.n_latents,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, epoch, folder='./trained_models_with_attention_400_50')   

            print('###fold number = ' + str(foldnumber))
            print('###weight_c = {:.3f}, weight_m = {:.3f}'.format(weight_c, weight_m))
            print('###highest_cindex = {:.3f}, MSE = {:.3f}, dice = {:.3f}'.format(highest_cindex, corresponding_mse, corresponding_dice))
            print('####')
