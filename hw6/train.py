#!/usr/bin/env python
# coding: utf-8

# %load train.py
import os
import pandas as pd

from utils.options import args
import utils.common as utils

from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR
from data import dataPreparer

import warnings, math

warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
checkpoint = utils.checkpoint(args)

from sklearn.metrics import confusion_matrix
import csv
import matplotlib.pyplot as plt


def main():
    start_epoch = 0
    best_acc = 0.0
 
    # Data loading
    print('=> Preparing data..')
 
    # data loader
    
    loader = dataPreparer.Data(args, 
                               data_path=args.src_data_path, 
                               label_path=args.src_label_path)
    
    data_loader = loader.loader_train
    data_loader_eval = loader.loader_valid
    
    data_predict = loader.loader_predict
    
    
    # Create model
    print('=> Building model...')

    # load training model
    model = import_module(f'model.{args.arch}').__dict__[args.model]().to(device)

    # Load pretrained weights
    if args.pretrained:
 
        ckpt = torch.load(os.path.join(checkpoint.ckpt_dir, args.source_file), map_location = device)
        state_dict = ckpt['state_dict']

        model.load_state_dict(state_dict)
        model = model.to(device)
        
    if args.inference_only:
        acc = inference(args, data_loader_eval, model, args.output_file)
        print(f'Test acc {acc:.3f}\n')
        return
    torch.manual_seed(42)
    param = [param for name, param in model.named_parameters()]
    
    #optimizer = optim.SGD(param, lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    optimizer = optim.AdamW(param, lr = args.lr, weight_decay = args.weight_decay)
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma = args.lr_gamma)
    
    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step(epoch)
        
        train(args, data_loader, model, optimizer, epoch)
        
        valid_acc = valid(args, data_loader_eval, model)
        
        is_best = best_acc < valid_acc
        best_acc = max(best_acc, valid_acc)
        if is_best:
            epo = epoch
        state = {
            'state_dict': model.state_dict(),
            
            'optimizer': optimizer.state_dict(),
            
            'scheduler': scheduler.state_dict(),
            
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)
    print(epo)    
    print(f'Best acc: {best_acc:.3f}\n')
        

    
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.xlabel('Step (K)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('fig.png')
    inference(args, data_predict, model, args.output_file)
    
    print(model)
    confusion(args, data_loader_eval, model)
  
       
def train(args, data_loader, model, optimizer, epoch):
    losses = utils.AverageMeter()

    acc = utils.AverageMeter()

    criterion = nn.CrossEntropyLoss()
    
    num_iterations = len(data_loader)
    
    # switch to train mode
    model.train()
        
    for i, (inputs, targets, _) in enumerate(data_loader, 1):

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # train
        output = model(inputs)
        loss = criterion(output, targets)

        # optimize cnn
        loss.backward()
        optimizer.step()

        ## train weights        
        losses.update(loss.item(), inputs.size(0))
        
        ## evaluate
        prec1, _ = utils.accuracy(output, targets, topk = (1, 5))
        acc.update(prec1[0], inputs.size(0))

        
        if i % args.print_freq == 0:     
            print(
                'Epoch[{0}]({1}/{2}): \n'
                'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                'Train acc {acc.val:.3f} ({acc.avg:.3f})\n'.format(
                epoch, i, num_iterations, 
                train_loss = losses,
                acc = acc))
        
            training_loss.append(losses.avg)
                
                
                
                
def valid(args, loader_valid, model):
    losses = utils.AverageMeter()
    acc = utils.AverageMeter()

    criterion = nn.CrossEntropyLoss()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets, datafile) in enumerate(loader_valid, 1):
            
            inputs = inputs.to(device)
            targets = targets.to(device)
             
            preds = model(inputs)
            loss = criterion(preds, targets)
            
            # image classification results
            prec1, _ = utils.accuracy(preds, targets, topk = (1, 5))
            
            losses.update(loss.item(), inputs.size(0))
            acc.update(prec1[0], inputs.size(0))
            
    validation_loss.append(losses.avg)
    print(f'Test acc {acc.avg:.3f}\n')

    return acc.avg
    

def inference(args, loader_valid, model, output_file_name):

    outputs = []
    datafiles = []
    count = 1
    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets, datafile) in enumerate(loader_valid, 1):
            
            inputs = inputs.to(device)
            targets = targets.to(device)
          
            preds = model(inputs)
            
            _, output = preds.topk(1, 1, True, True)
            
            outputs.extend(list(output.reshape(-1).cpu().detach().numpy()))
            
            datafiles.extend(list(datafile))
            
            count += inputs.size(0)
    

    output_file = dict()
    output_file['image_name'] = datafiles
    output_file['label'] = outputs
    
    output_file = pd.DataFrame.from_dict(output_file)
    output_file.to_csv(output_file_name, index = False)
    
    
def confusion(args, loader_valid, model):

    outputs = []
    datafiles = []
    count = 1
    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets, datafile) in enumerate(loader_valid, 1):
            
            inputs = inputs.to(device)
            targets = targets.to(device)
          
            preds = model(inputs)
        
            # image classification results
            prec1, _ = utils.accuracy(preds, targets, topk = (1, 5))
            
            _, output = preds.topk(1, 1, True, True)
            
            outputs.extend(list(output.reshape(-1).cpu().detach().numpy()))
            
            datafiles.extend(list(datafile))
            
            count += inputs.size(0)
    
    with open('../digit/valid.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        label = []
        i = 0
        for row in rows:
            if(i>0):
                label.append(int(row[1]))
            i+=1
    conf_matrix = confusion_matrix(label, outputs)        
    print(conf_matrix)
    # Print the confusion matrix using Matplotlib

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig('confusion_matrix.png')
    
if __name__ == '__main__':
    training_loss = []
    validation_loss = []

    main()
