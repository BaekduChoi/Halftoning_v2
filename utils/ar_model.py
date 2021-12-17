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

class ar_model :
    def __init__(self,json_dir,cuda=True,depth=6,depth2=2,ndf=16,ar_mid_ndf=64) :

        torch.autograd.set_detect_anomaly(True)

        self.params = read_json(json_dir)
        self.device = torch.device('cuda') if cuda else torch.device('cpu')

        self.ksize = 5

        self.netFE = SimpleNetDenseINDepth3(in_ch=2,out_nch=1,ndf=16,depth=depth)
        fe_load = torch.load('checkpts_ndf16/epoch80.ckp')
        self.netFE.load_state_dict(fe_load['modelG_state_dict'])

        self.train_fe = False

        if not self.train_fe :
            self.netFE.blockout = nn.Sequential(
                nn.Conv2d(16,16,7,1,3,padding_mode='circular'),
                nn.ReLU(True),
                nn.Conv2d(16,16,7,1,3,padding_mode='circular'),
                nn.ReLU(True),
                nn.Conv2d(16,ndf,1)
            )
        else :
            self.netFE.blockout = nn.Conv2d(16,ndf,1)

        if not self.train_fe :
            for _p in self.netFE.parameters() :
                _p.requires_grad_(False)
            for _p in self.netFE.blockout.parameters() :
                _p.requires_grad_(True)

        self.netPx = ARPrior(feature_input_dim=ndf,out_ch=1,depth=depth2,mid_dim=ar_mid_ndf,ksize=self.ksize)

        self.netFE = self.netFE.to(self.device)
        self.netPx = self.netPx.to(self.device)

        self.depth2 = depth2

        self.clip_grad = True

        self.random_screen = True

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
        params_list = list(self.netPx.parameters())
        if self.train_fe :
            params_list += list(self.netFE.parameters())
        else :
            params_list += list(self.netFE.blockout.parameters())
        self.optimizer = optim.Adam(params_list,lr=self.lr,betas=self.betas,amsgrad=True)

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
                    imgsR = data['halftone']
                    inputS = data['screened']
                    inputG = inputG.to(self.device)
                    inputS = inputS.to(self.device)
                    imgsR = imgsR.to(self.device)

                    output, loss_GT = self.fit(inputG,inputS,imgsR)
                    
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
                val_loss = self.val(valloader,self.val_path,epoch)
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
    
    def test_final(self,use_cpu=False) :
        self.loadckp_test()

        testloader = create_test_dataloaders(self.params)
        test_path = self.params["solver"]["testpath"]
        if test_path[-1] != '/' :
            test_path += '/'

        if not os.path.isdir(test_path) :
            os.mkdir(test_path)

        self.test(testloader,test_path,save_scr=True,use_cpu=use_cpu)
    
    def loadckp_test(self) :
        self.ckp_load = torch.load(self.params['solver']['ckp_path'])
        self.netPx.load_state_dict(self.ckp_load['modelPx_state_dict'])
        self.netFE.load_state_dict(self.ckp_load['modelFE_state_dict'])

    def test(self,testloader,test_dir,save_scr=True,use_cpu=False) :
        # with torch.no_grad() :
        with torch.inference_mode() :
            # self.netFE.eval()
            # self.netPx.eval()
            count = 0
            with tqdm(total=len(testloader),\
                    desc='Testing.. ',miniters=1) as t:
                for ii,data in enumerate(testloader) :
                    inputG = data['img']
                    inputG = inputG.to(self.device)

                    inputS = data['screened']
                    inputS = inputS.to(self.device)

                    features,_ = self.netFE(torch.cat([inputG,inputS],dim=1))
                    
                    img_size1,img_size2 = inputG.shape[2], inputG.shape[3]
                    bsize = inputG.shape[0]
                    # outputs = torch.zeros_like(inputG)
                    # outputs = inputS.detach().clone() # initialize using screen

                    # psize = self.psize

                    # combined = torch.cat([features,outputs],dim=1)
                    
                    # outputs = torch.zeros_like(inputG)
                    if use_cpu :
                        outputs = inputS.detach().cpu()
                        features = features.detach().cpu()
                        self.netPx = self.netPx.to(torch.device('cpu'))
                    else :
                        outputs = inputS.detach().clone()
                        features = features.detach()
                    pad = self.ksize//2
                    outputs = F.pad(outputs,(pad,pad,pad,pad)) # zero-pad outputs
                    # print(outputs.shape)

                    # features = features.detach().cpu()
                    # outputs = outputs.cpu()

                    for j in range(outputs.shape[0]) :
                        for y in range(img_size1) :
                            for x in range(img_size2) :
                                
                                features_curr = features[j,:,y,x].squeeze()
                                outputs_prev = outputs[j,0,y,x:x+self.ksize].squeeze()
                                for ppp in range(1,pad) :
                                    outputs_prev = torch.cat([outputs_prev,outputs[j,0,y+ppp,x:x+self.ksize].squeeze()])
                                outputs_prev = torch.cat([outputs_prev,outputs[j,0,y+pad,x:x+pad].squeeze()])
                                
                                probs = self.netPx(torch.cat([features_curr,outputs_prev]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
                                prob = probs.squeeze()
                                outputs[j,0,y+pad,x+pad] = torch.bernoulli(prob)
                        outputs_curr = outputs[j,0,pad:img_size1+pad,pad:img_size2+pad]

                        imgR = torch.zeros([img_size1,img_size2],dtype=torch.float32)
                        imgR[:,:] = outputs_curr.squeeze()
                        imgR = imgR.detach().numpy() if use_cpu else imgR.detach().cpu().numpy()
                        imgR = np.clip(imgR,0,1)
                        imgBGR = (255*imgR).astype('uint8')
                        imname = test_dir+str(count+1)+'.png'
                        cv2.imwrite(
                            imname,imgBGR)

                        if save_scr :
                            imgS = torch.zeros([img_size1,img_size2],dtype=torch.float32)
                            imgS[:,:] = inputS[j,0,:,:].squeeze()
                            imgS = imgS.cpu().numpy()
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

                    features,_ = self.netFE(torch.cat([inputG,inputS],dim=1))

                    pad_pix_nums = self.ksize*(self.ksize//2)+self.ksize//2 # assumes ksize is odd
                    imgsH_pad = torch.zeros((features.shape[0],pad_pix_nums,features.shape[2]+self.ksize//2,features.shape[3]+self.ksize)).to(self.device)
                    for i in range(pad_pix_nums) :
                        xshift = self.ksize//2-i%(self.ksize)+self.ksize//2
                        yshift = self.ksize//2-i//(self.ksize)
                        imgsH_pad[:,i,yshift:yshift+features.shape[2],xshift:xshift+features.shape[3]] = imgsH.squeeze()
                    imgsH_pad = imgsH_pad[:,:,:features.shape[2],self.ksize//2:self.ksize//2+features.shape[3]].detach()
                    
                    img_size1,img_size2 = inputG.shape[2], inputG.shape[3]
                    bsize = inputG.shape[0]

                    outputs_loss = self.netPx(torch.cat([features,imgsH_pad],dim=1))

                    loss = self.GT_loss(outputs_loss,imgsH).item()

                    running_loss += loss/len(testloader)

                    if ii < 4 : # generate 4 images for visual inspection
                        # outputs = torch.zeros_like(inputG)
                        outputs = inputS.detach().clone()
                        pad = self.ksize//2
                        outputs = F.pad(outputs,(pad,pad,pad,pad)) # zero-pad outputs

                        for j in range(outputs.shape[0]) :
                            for y in range(img_size1) :
                                for x in range(img_size2) :
                                    
                                    features_curr = features[j,:,y,x].squeeze()
                                    outputs_prev = outputs[j,0,y,x:x+self.ksize].squeeze()
                                    for ppp in range(1,pad) :
                                        outputs_prev = torch.cat([outputs_prev,outputs[j,0,y+ppp,x:x+self.ksize].squeeze()])
                                    outputs_prev = torch.cat([outputs_prev,outputs[j,0,y+pad,x:x+pad].squeeze()])
                                    
                                    probs = self.netPx(torch.cat([features_curr,outputs_prev]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
                                    prob = probs.squeeze()
                                    outputs[j,0,y+pad,x+pad] = torch.bernoulli(prob)
                            outputs_curr = outputs[j,0,pad:img_size1+pad,pad:img_size2+pad]

                            imgR = torch.zeros([img_size1,img_size2],dtype=torch.float32)
                            imgR[:,:] = outputs_curr.squeeze()
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
                'modelPx_state_dict':self.netPx.state_dict(),
                'modelFE_state_dict':self.netFE.state_dict(),
                'optimizer_state_dict':self.optimizer.state_dict(),
                'scheduler_state_dict':self.scheduler.state_dict(),
                'loss':self.running_loss
            },path)

    
    def loadckp(self) :
        self.ckp_load = torch.load(self.params['solver']['ckp_path'])
        self.start_epochs = self.ckp_load['epoch']
        self.netPx.load_state_dict(self.ckp_load['modelPx_state_dict'])
        self.netFE.load_state_dict(self.ckp_load['modelFE_state_dict'])
        self.optimizer.load_state_dict(self.ckp_load['optimizer_state_dict'])
        self.scheduler.load_state_dict(self.ckp_load['scheduler_state_dict'])

        loss_load = self.ckp_load['loss']

        print('Resumed training - epoch %d with loss = %.4f'%(self.start_epochs+1,loss_load))

    def fit(self,inputG,inputS,imgsH) :

        features,_ = self.netFE(torch.cat([inputG,inputS],dim=1))

        pad_pix_nums = self.ksize*(self.ksize//2)+self.ksize//2 # assumes ksize is odd
        imgsH_pad = torch.zeros((features.shape[0],pad_pix_nums,features.shape[2]+self.ksize//2,features.shape[3]+self.ksize)).to(self.device)
        for i in range(pad_pix_nums) :
            xshift = self.ksize//2-i%(self.ksize)+self.ksize//2
            yshift = self.ksize//2-i//(self.ksize)
            imgsH_pad[:,i,yshift:yshift+features.shape[2],xshift:xshift+features.shape[3]] = imgsH.squeeze()
        imgsH_pad = imgsH_pad[:,:,:features.shape[2],self.ksize//2:self.ksize//2+features.shape[3]].detach()

        outputs = self.netPx(torch.cat([features,imgsH_pad],dim=1))

        # loss = self.GT_loss(outputs,imgsH)
        pad = self.ksize//2
        loss = self.GT_loss(outputs[:,:,pad:-pad,pad:-pad],imgsH[:,:,pad:-pad,pad:-pad])
        
        # generator weight update
        # for the generator, all the loss terms are used
        self.optimizer.zero_grad()

        loss.backward()

        if self.clip_grad :
            torch.nn.utils.clip_grad_norm_(self.netPx.parameters(),1.0)
            if self.train_fe :
                torch.nn.utils.clip_grad_norm_(self.netFE.parameters(),1.0)
            else :
                torch.nn.utils.clip_grad_norm_(self.netFE.blockout.parameters(),1.0)

        # backpropagation for generator and encoder
        self.optimizer.step()
        # check only the L1 loss with GT colorization for the fitting procedure
        self.running_loss += loss.item()/self.num_batches

        return outputs.detach(), loss.item()
    
    def test_pad(self) :
        imgsH = torch.tensor([[[[1,2,3,4,5],
            [6,7,8,9,10],
            [11,12,13,14,15],
            [16,17,18,19,20],
            [21,22,23,24,25]]]])
        self.ksize = 5

        pad_pix_nums = self.ksize*(self.ksize//2)+self.ksize//2 # assumes ksize is odd
        imgsH_pad = torch.zeros((1,pad_pix_nums,5+self.ksize//2,5+self.ksize))
        for i in range(pad_pix_nums) :
            xshift = self.ksize//2-i%(self.ksize)+self.ksize//2
            yshift = self.ksize//2-i//(self.ksize)
            imgsH_pad[:,i,yshift:yshift+5,xshift:xshift+5] = imgsH.squeeze()
        imgsH_pad = imgsH_pad[:,:,:5,self.ksize//2:self.ksize//2+5]
        # print(imgsH_pad)

        pad = self.ksize//2
        outputs = F.pad(imgsH,(pad,pad,pad,pad))
        x = 2
        y = 2
        j = 0
        outputs_prev = outputs[j,0,y,x:x+self.ksize].squeeze()
        for ppp in range(1,pad) :
            outputs_prev = torch.cat([outputs_prev,outputs[j,0,y+ppp,x:x+self.ksize].squeeze()])
        outputs_prev = torch.cat([outputs_prev,outputs[j,0,y+pad,x:x+pad].squeeze()])
        print(outputs_prev)

        features = torch.randn((5))
        print(torch.cat([features,outputs_prev]))