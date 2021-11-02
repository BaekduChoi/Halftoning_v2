import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))

from torch import nn
import torch
from torch import optim
from torch.nn import functional as F
from abc import ABC
from tqdm import tqdm
import numpy as np
import cv2
import argparse

from blocks import *
from network import *
from misc import *
from losses import *
import pickle

class pixcnn_model :
    def __init__(self,json_dir,cuda=True,depth=6,depth2=4,ndf=32) :

        torch.autograd.set_detect_anomaly(True)

        self.params = read_json(json_dir)
        self.device = torch.device('cuda') if cuda else torch.device('cpu')

        self.netFE = FeatureExtractor(in_ch=2,ndf=ndf,depth=depth,padding_mode='zeros')
        self.netPx = PixCNNPrior(feature_input_dim=ndf,out_ch=1,depth=depth2,padding_mode='zeros')

        self.netFE = self.netFE.to(self.device)
        self.netPx = self.netPx.to(self.device)

        self.depth2 = depth2

    def getparams(self) :
        # reading the hyperparameter values from the json file
        self.lr = self.params['solver']['learning_rate']
        self.lr_step = self.params['solver']['lr_step']
        self.lr_gamma = self.params['solver']['lr_gamma']
        self.lambda_hvs = self.params['solver']['lambda_hvs'] # may be used later
        self.beta1 = self.params['solver']['beta1']
        self.beta2 = self.params['solver']['beta2']
        self.betas = (self.beta1,self.beta2)
        self.batch_size = self.params['datasets']['train']['batch_size']
    
    def getopts(self) :
        # set up the optimizers and schedulers
        self.optimizer = optim.Adam(list(self.netFE.parameters())+list(self.netPx.parameters()),lr=self.lr,betas=self.betas,amsgrad=True)

        self.lr_sche_ftn = lambda epoch : 1.0 if epoch < self.lr_step else (self.lr_gamma)**(epoch-self.lr_step)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,self.lr_sche_ftn)

    def train(self) :
        trainloader, valloader = create_dataloaders(self.params)

        # GT loss
        self.GT_loss = nn.BCELoss()

        self.inittrain()
        # num_batches is saved for normalizing the running loss
        self.num_batches = len(trainloader)

        # starting iteration
        for epoch in range(self.start_epochs,self.epochs) :
            print('Epoch = '+str(epoch+1))

            # training part of the iteration
            self.running_loss = 0.0

            # tqdm setup is borrowed from SRFBN github
            # https://github.com/Paper99/SRFBN_CVPR19
            with tqdm(total=len(trainloader),\
                    desc='Epoch: [%d/%d]'%(epoch+1,self.epochs),miniters=1) as t:
                for i,data in enumerate(trainloader) :
                    # inputG = input image in grayscale
                    inputG = data['img']
                    imgsH = data['halftone']
                    inputS = data['screened']
                    inputG = inputG.to(self.device)
                    inputS = inputS.to(self.device)
                    imgsH = imgsH.to(self.device)

                    output, loss_GT = self.fit(inputG,inputS,imgsH)
                    
                    # tqdm update
                    t.set_postfix_str('G loss : %.4f'%(loss_GT))
                    t.update()
                    
            # print the running loss     
            print('Finished training for epoch %d, running loss = %.4f'%(epoch+1,self.running_loss))
            self.train_losses.append(self.running_loss)
            
            # validation is tricky for GANs - what to use for validation?
            # since no quantitative metric came to mind, I am just saving validation results
            # visually inspecting them helps finding issues with training
            # the validation results are saved in validation path
            if valloader is not None :
                val_loss = self.val(valloader,self.val_path,4,epoch)
                self.val_losses.append(val_loss)

            self.scheduler.step()
            
            print(self.scheduler.get_last_lr())

            self.saveckp(epoch)

        
    def inittrain(self) :
        self.getparams()
        self.getopts()

        # reading more hyperparameters and checkpoint saving setup
        # head_start determines how many epochs the generator will head-start learning
        self.epochs = self.params['solver']['num_epochs']
        self.save_ckp_step = self.params['solver']['save_ckp_step']
        self.pretrained_path = self.params['solver']['pretrained_path']
        self.val_path = self.params['solver']['val_path']

        # hvs
        hvs = HVS().getHVS().astype(np.float32)
        self.hvs = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(hvs),0),0).to(self.device)

        if self.val_path[-1] != '/':
            self.val_path += '/'

        if not os.path.isdir(self.pretrained_path) :
            os.mkdir(self.pretrained_path)
        
        if not os.path.isdir(self.val_path) :
            os.mkdir(self.val_path)

        # code for resuming training
        # if pretrain = False, the training starts from scratch as expected
        # otherwise, the checkpoint is loaded back and training is resumed
        # for the checkpoint saving format refer to the end of the function
        self.start_epochs = 0
        self.pretrain = self.params['solver']['pretrain']

        if self.pretrain :
            self.loadckp()    
            print(self.scheduler.get_last_lr())
        
        if self.pretrain and os.path.exists(self.pretrained_path+'losses.pkl') :
            losses_saved = pickle.load(open(self.pretrained_path+'losses.pkl','rb'))
            self.train_losses = losses_saved[0]
            self.val_losses = losses_saved[1]
        else :
            self.train_losses = []
            self.val_losses = []
    
    def test_final(self) :
        self.loadckp_test()

        testloader = create_test_dataloaders(self.params)
        test_path = self.params["solver"]["testpath"]
        if test_path[-1] != '/' :
            test_path += '/'

        if not os.path.isdir(test_path) :
            os.mkdir(test_path)

        self.test(testloader,test_path,save_scr=True)
    
    def loadckp_test(self) :
        self.ckp_load = torch.load(self.params['solver']['ckp_path'])
        self.netFE.load_state_dict(self.ckp_load['modelFE_state_dict'])
        self.netPx.load_state_dict(self.ckp_load['modelPx_state_dict'])

    def test(self,testloader,test_dir,save_scr=True) :
        with torch.no_grad() :
            count = 0
            with tqdm(total=len(testloader),\
                    desc='Testing.. ',miniters=1) as t:
                for ii,data in enumerate(testloader) :
                    inputG = data['img']
                    inputG = inputG.to(self.device)

                    inputS = data['screened']
                    inputS = inputS.to(self.device)

                    features = self.netFE(torch.cat([inputG,inputS],dim=1))
                    
                    img_size1,img_size2 = inputG.shape[2], inputG.shape[3]
                    bsize = inputG.shape[0]
                    outputs = torch.zeros_like(inputG)

                    psize = 3+4*(self.depth2-1)
                    
                    for j in range(outputs.shape[0]) :
                        for y in range(img_size1) :
                            for x in range(img_size2) :
                                start_y = max(y-psize//2,0)
                                start_x = max(x-psize//2,0)
                                end_y = min(y+psize//2+1,img_size1)
                                end_x = min(x+psize//2+1,img_size2)

                                features_curr = features[j,:,start_y:end_y,start_x:end_x].unsqueeze(0)
                                outputs_curr = outputs[j,:,start_y:end_y,start_x:end_x].unsqueeze(0)
                                probs = self.netPx(features_curr,outputs_curr)
                                prob = probs[j,0,y-start_y,x-start_x]
                                outputs[j,0,y,x] = torch.bernoulli(prob)

                        imgR = torch.zeros([img_size1,img_size2],dtype=torch.float32)
                        imgR[:,:] = outputs[j,:,:].squeeze()
                        imgR = imgR.detach().numpy()
                        imgR = np.clip(imgR,0,1)
                        imgBGR = (255*imgR).astype('uint8')
                        imname = test_dir+str(count+1)+'.png'
                        cv2.imwrite(
                            imname,imgBGR)

                        if save_scr :
                            imgS = torch.zeros([img_size1,img_size2],dtype=torch.float32)
                            imgS[:,:] = inputS[j,0,:,:].squeeze()
                            imgS = imgS.numpy()
                            sname = test_dir+str(count+1)+'_scr.png'
                            cv2.imwrite(sname,(255*imgS).astype('uint8'))
                        
                        count += 1
                    # tqdm update
                    t.update()
    
    def val(self,testloader,test_dir,epoch=None,save_scr=True) :
        with torch.no_grad() :
            count = 0
            running_loss = 0.
            with tqdm(total=len(testloader),\
                    desc='Validating.. ',miniters=1) as t:
                for ii,data in enumerate(testloader) :
                    inputG = data['img']
                    inputG = inputG.to(self.device)

                    inputS = data['screened']
                    inputS = inputS.to(self.device)

                    imgsH = data['halftone'].to(self.device)

                    features = self.netFE(torch.cat([inputG,inputS],dim=1))
                    
                    img_size1,img_size2 = inputG.shape[2], inputG.shape[3]
                    bsize = inputG.shape[0]

                    outputs = self.netPx(features,imgsH)

                    loss = self.GT_loss(outputs,imgsH)

                    running_loss += loss/len(testloader)

                    if ii < 4 : # generate 4 images for visual inspection
                        outputs = torch.zeros_like(inputG)

                        psize = 3+4*(self.depth2-1)
                        for j in range(outputs.shape[0]) :
                            for y in range(img_size1) :
                                for x in range(img_size2) :
                                    start_y = max(y-psize//2,0)
                                    start_x = max(x-psize//2,0)
                                    end_y = min(y+psize//2+1,img_size1)
                                    end_x = min(x+psize//2+1,img_size2)

                                    features_curr = features[j,:,start_y:end_y,start_x:end_x].unsqueeze(0)
                                    outputs_curr = outputs[j,:,start_y:end_y,start_x:end_x].unsqueeze(0)
                                    probs = self.netPx(features_curr,outputs_curr)
                                    prob = probs[j,0,y-start_y,x-start_x]
                                    outputs[j,0,y,x] = torch.bernoulli(prob)

                            imgR = torch.zeros([img_size1,img_size2],dtype=torch.float32)
                            imgR[:,:] = outputs[j,:,:].squeeze()
                            imgR = imgR.detach().numpy()
                            imgR = np.clip(imgR,0,1)
                            imgBGR = (255*imgR).astype('uint8')
                            imname = test_dir+str(count+1)+'_epoch'+str(epoch+1)+'.png' if epoch != None else test_dir+str(count+1)+'.png'
                            cv2.imwrite(
                                imname,imgBGR)

                            if save_scr :
                                imgS = torch.zeros([img_size1,img_size2],dtype=torch.float32)
                                imgS[:,:] = inputS[j,0,:,:].squeeze()
                                imgS = imgS.numpy()
                                sname = test_dir+str(count+1)+'_scr.png'
                                cv2.imwrite(sname,(255*imgS).astype('uint8'))
                            
                            count += 1
                    t.update()
        return running_loss
    
    def saveckp(self,epoch) :
        pickle.dump([self.train_losses,self.val_losses],open(self.pretrained_path+'losses.pkl','wb'))
        if (epoch+1)%self.save_ckp_step == 0 :
            path = self.pretrained_path+'/epoch'+str(epoch+1)+'.ckp'
            torch.save({
                'epoch':epoch+1,
                'modelFE_state_dict':self.netFE.state_dict(),
                'modelPx_state_dict':self.netPx.state_dict(),
                'optimizer_state_dict':self.optimizer.state_dict(),
                'scheduler_state_dict':self.scheduler.state_dict(),
                'loss':self.running_loss
            },path)

    
    def loadckp(self) :
        self.ckp_load = torch.load(self.params['solver']['ckp_path'])
        self.start_epochs = self.ckp_load['epoch']
        self.netFE.load_state_dict(self.ckp_load['modelFE_state_dict'])
        self.netPx.load_state_dict(self.ckp_load['modelPx_state_dict'])
        self.optimizer.load_state_dict(self.ckp_load['optimizer_state_dict'])
        self.scheduler.load_state_dict(self.ckp_load['scheduler_state_dict'])

        loss_load = self.ckp_load['loss']

        print('Resumed training - epoch %d with loss = %.4f'%(self.start_epochs+1,loss_load))

    def fit(self,inputG,inputS,imgsH) :

        features = self.netFE(torch.cat([inputG,inputS],dim=1))
        outputs = self.netPx(features,imgsH)

        loss = self.GT_loss(outputs,imgsH)
        
        # generator weight update
        # for the generator, all the loss terms are used
        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.netFE.parameters(),1.0)
        torch.nn.utils.clip_grad_norm_(self.netPx.parameters(),1.0)

        # backpropagation for generator and encoder
        self.optimizer.step()
        # check only the L1 loss with GT colorization for the fitting procedure
        self.running_loss += loss.item()/self.num_batches

        return outputs.detach(), loss.item()