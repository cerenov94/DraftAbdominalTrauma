import os

import numpy as np
import torch
import torchio as tio

from torch.utils.data import DataLoader
from configs import Config
from time import time
import datetime as dtime
from compet_metric_and_other_utils import log_score
import gc
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:512"
pos_weight = torch.tensor([2.0]).to(device)

def train(model,train_dataset,valid_dataset,resume = False):
    gc.collect()
    torch.cuda.empty_cache()
    ## loss for binary
    BCE = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    CE = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    last_epoch = 0
    best_log_score = float('inf')
    best_loss = float('inf')
    best_valid_score = float('inf')
    model.to(device)
    if resume:
        print('Loading last checkpoint...')
        model.load_state_dict(torch.load('logs/last_weights.pth'))
        optimizer.load_state_dict(torch.load('logs/optimizer_dict.pth'))
        best_loss = Config.ckp['best_loss']
        best_log_score = Config.ckp['best_log_score']
        best_valid_score = Config.valid_ckp['best_val_score']
        last_epoch = Config.ckp['epoch']
    print(f'Starting train from: Epoch: {last_epoch} | Best train loss : {best_loss} | Best valid score : {best_valid_score}')


        ## add loading best valid score


    train_dataloader = DataLoader(train_dataset, batch_size=1,shuffle=True, num_workers=2,pin_memory=True)


    for epoch in range(last_epoch,1000):
        train_loss = 0

        start_time = time()
        model.train()
        y_true = []
        y_pred = []


        for x,targets in train_dataloader:
            x = x.type(torch.float32)
            targets = targets.type(torch.float32)
            x,targets = x.to(device),targets.to(device)

            prediction = model(x)



            loss = (BCE(prediction[:,0],targets[:,0]) +
                    BCE(prediction[:,1],targets[:,1]) +
                    CE(prediction[:,2:5],targets[:,2:5]) +
                    CE(prediction[:,5:8],targets[:,5:8]) +
                    CE(prediction[:,8:11],targets[:,8:11]))
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            # saving loss and compet metrci
            train_loss += loss.item()
            prediction = torch.concatenate((torch.sigmoid(prediction[:, 0]).reshape(-1, 1),
                                            torch.sigmoid(prediction[:, 1]).reshape(-1, 1),
                                            torch.softmax(prediction[:, 2:5], dim=1),
                                            torch.softmax(prediction[:, 5:8], dim=1),
                                            torch.softmax(prediction[:, 8:11], dim=1),
                                            # torch.sigmoid(prediction[:,-1]).reshape(-1,1)
                                            ), dim=1).detach().cpu().numpy()
            y_pred.append(prediction)
            y_true.append(targets.detach().cpu().numpy())


        score = log_score(y_true,y_pred)
        scheduler.step()

        ## score and loss for epoch
        train_loss = train_loss/len(train_dataloader)

        duration = str(dtime.timedelta(seconds = time() - start_time))[:7]
        ## showing results
        print(f"EPOCH : {epoch+1} | Duration {duration} | TRAIN LOSS: {train_loss:.4f} | TRAIN SCORE: {score:.4f}")
        #if epoch % 5 == 0:


        # saving model
        if train_loss < best_loss or score < best_log_score or (epoch+1) % 5 == 0:
            if train_loss < best_loss:
                best_loss = train_loss
            if score < best_log_score:
                best_log_score = score
            torch.save(model.state_dict(),'logs/last_weights.pth')
            torch.save(optimizer.state_dict(),'logs/optimizer_dict.pth')
            torch.save({'epoch': epoch+1,
                        'best_loss': best_loss,
                        'best_log_score': best_log_score,
                        },f'logs/additional_info.pth')
            print('Train logs saved.')
        # validation step
        if (epoch + 1) % 5 == 0:
            valid_score = validation_step(model, valid_dataset)
            if valid_score < best_valid_score:
                best_valid_score = valid_score
            # saving best validation score
            torch.save(model.state_dict(),f'logs/validation_best_score.pth')
            torch.save(optimizer.state_dict(),f'logs/optimizer_valid_dict.pth')
            torch.save({'best_val_score': best_valid_score},f'logs/best_valid_score.pth')
            print('Validation logs saved')


def validation_step(model,valid_dataset):
    start_time = time()
    dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size=1,pin_memory=True)
    y_true = []
    y_pred = []
    model.eval()
    with torch.inference_mode():
        for x,targets in dataloader:
            x = x.type(torch.float32)
            targets = targets.type(torch.float32)
            x,targets = x.to(device),targets.to(device)

            prediction = model(x)
            prediction_probas = torch.concatenate((torch.sigmoid(prediction[:, 0]).reshape(-1, 1),
                                                   torch.sigmoid(prediction[:, 1]).reshape(-1, 1),
                                                   torch.softmax(prediction[:, 2:5], dim=1),
                                                   torch.softmax(prediction[:, 5:8], dim=1),
                                                   torch.softmax(prediction[:, 8:11], dim=1)), dim=1).detach().cpu().numpy()
            y_true.append(targets.detach().cpu().numpy())
            y_pred.append(prediction_probas)
    valid_score = log_score(y_true,y_pred)
    #valid_score /= len(dataloader)
    duration = str(dtime.timedelta(seconds=time() - start_time))[:7]
    ## showing results
    print(f"Validation time duration : {duration} | VALID SCORE: {valid_score:.4f}")
    return valid_score
