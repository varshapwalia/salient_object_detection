import torch
from collections import OrderedDict
from torch.optim import Adam, SGD
from torch.autograd import Variable
from model import build_model, weights_init
import numpy as np
import os
import torchvision.utils as vutils
import cv2
import torch.nn.functional as F
import time
import os

# Initialize an empty ordered dictionary p
EPSILON = 1e-8
p = OrderedDict()

from dataset import get_loader
base_model_cfg = 'resnet'
p['lr_bone'] = 5e-5                             # Learning rate for resnet: 5e-5
p['lr_branch'] = 0.025                          # Learning rate for the other branches
p['wd'] = 0.0005                                # Weight decay
p['momentum'] = 0.90                            # Momentum
lr_decay_epoch = [15, 24]                       # [6, 9], now x3 #15
nAveGrad = 10                                   # Update the weights once in 'nAveGrad' forward passes
showEvery = 50                                  # Frequency that the training progress will be displayed
tmp_path = 'tmp_see'                            # Path for temporary image outputs

# For training and testing
class Solver(object):
    def __init__(self, train_loader, test_loader, config, save_fold=None):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.save_fold = save_fold
        self.build_model()
        if self.config.pre_trained: self.net.load_state_dict(torch.load(self.config.pre_trained))
        if config.mode == 'train':
            self.log_output = open("%s/logs/log.txt" % config.save_fold, 'w')
        else:
            print('Loading pre-trained model from %s...' % self.config.model)
            if self.config.cuda:
                self.net_bone.load_state_dict(torch.load(self.config.model))
            else:
                self.net_bone.load_state_dict(torch.load(self.config.model, map_location=torch.device('cpu')))
            self.net_bone.eval()

    # Method to build the neural network model
    def build_model(self):
        self.net_bone = build_model(base_model_cfg)
        # check for CUDA enabled and move model to GPU for faster performance
        if self.config.cuda:
            self.net_bone = self.net_bone.cuda()  
        self.net_bone.eval()    # use_global_stats = True
        # initialize the weights of the model
        self.net_bone.apply(weights_init)
        if self.config.mode == 'train':
            if self.config.load_bone == '':
                if base_model_cfg == 'resnet':
                    self.net_bone.base.load_state_dict(torch.load(self.config.resnet))      #loads pre-trained dictionary of ResNet backbone
            if self.config.load_bone != '': self.net_bone.load_state_dict(torch.load(self.config.load_bone))

        self.lr_bone = p['lr_bone']
        self.lr_branch = p['lr_branch']
        # construct Adam optimizer for updating parameters of the backbone model
        self.optimizer_bone = Adam(filter(lambda p: p.requires_grad, self.net_bone.parameters()), lr=self.lr_bone, weight_decay=p['wd'])

    # Testing phase
    def test(self, test_mode=0):
        EPSILON = 1e-8
        img_num = len(self.test_loader)
        time_t = 0.0
        name_t = 'EGNet_ResNet50\\'

        if not os.path.exists(os.path.join(self.save_fold, name_t)):             
            os.mkdir(os.path.join(self.save_fold, name_t))
        for i, data_batch in enumerate(self.test_loader):
            self.config.test_fold = self.save_fold
            print(self.config.test_fold)
            images_, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            
            with torch.no_grad():   # temporarily disables gradient calculation                            
                
                images = Variable(images_)
                if self.config.cuda:
                    images = images.cuda()
                print(images.size())
                time_start = time.time()
                up_edge, up_sal, up_sal_f = self.net_bone(images)       # forward pass images through the nnm
                if self.config.cuda:
                    torch.cuda.synchronize()
                time_end = time.time()
                print(time_end - time_start)
                time_t = time_t + time_end - time_start                              
                pred = np.squeeze(torch.sigmoid(up_sal_f[-1]).cpu().data.numpy())       # extract final saliency map         
                multi_fuse = 255 * pred     # scale saliency map values to [0, 255]
      
                # # Debug - Check if the prediction is valid
                # if multi_fuse is not None:
                #     print("Image loaded successfully!")
                #     print(os.path.join(self.config.test_fold, name_t, os.path.basename(name)[:-4] + '.png'))
                #     # Display the image
                #     cv2.imshow("Image", multi_fuse)
                #     cv2.waitKey(0)  # Wait for a key press to close the window
                #     cv2.destroyAllWindows()  # Close all OpenCV windows
                # else:
                #     print("Failed to load image.")
                    
                cv2.imwrite(os.path.join(self.config.test_fold, name_t, os.path.basename(name)[:-4] + '.png'), multi_fuse)     # save map as an image
          
        print("--- %s seconds ---" % (time_t))
        print('Test Done!')

   
    # Training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size     # total number of iterations per epoch
        aveGrad = 0
        F_v = 0
        if not os.path.exists(tmp_path): 
            os.mkdir(tmp_path)
        
        for epoch in range(self.config.epoch):                                 
            r_edge_loss, r_sal_loss, r_sum_loss= 0,0,0      # keep track of losses from each iteration
            self.net_bone.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_label, sal_edge = data_batch['sal_image'], data_batch['sal_label'], data_batch['sal_edge']       # extract each from current data batch
                if sal_image.size()[2:] != sal_label.size()[2:]:
                    print("Skip this batch")
                    continue
                sal_image, sal_label, sal_edge = Variable(sal_image), Variable(sal_label), Variable(sal_edge)
                if self.config.cuda: 
                    sal_image, sal_label, sal_edge = sal_image.cuda(), sal_label.cuda(), sal_edge.cuda()

                # pass salienct images through the network
                up_edge, up_sal, up_sal_f = self.net_bone(sal_image)
                
                # compute edge loss using binary cross=entropy loss function
                edge_loss = []
                for ix in up_edge:
                    edge_loss.append(bce2d_new(ix, sal_edge, reduction='sum'))
                edge_loss = sum(edge_loss) / (nAveGrad * self.config.batch_size)
                r_edge_loss += edge_loss.data
                
                # compute saliency losses using binary cross=entropy loss function
                sal_loss1= []
                sal_loss2 = []
                for ix in up_sal:
                    sal_loss1.append(F.binary_cross_entropy_with_logits(ix, sal_label, reduction='sum'))

                for ix in up_sal_f:
                    sal_loss2.append(F.binary_cross_entropy_with_logits(ix, sal_label, reduction='sum'))
                sal_loss = (sum(sal_loss1) + sum(sal_loss2)) / (nAveGrad * self.config.batch_size)
              
                r_sal_loss += sal_loss.data

                # total loss
                loss = sal_loss + edge_loss
                r_sum_loss += loss.data
                
                # computes gradients of the model with respect to the loss
                loss.backward()
                aveGrad += 1

                # update parameters using optimizer
                if aveGrad % nAveGrad == 0:
       
                    self.optimizer_bone.step()
                    self.optimizer_bone.zero_grad()           
                    aveGrad = 0

                # prints training progress at every 'showEvery' iterations
                if i % showEvery == 0:

                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Edge : %10.4f  ||  Sal : %10.4f  ||  Sum : %10.4f' % (
                        epoch, self.config.epoch, i, iter_num,  r_edge_loss*(nAveGrad * self.config.batch_size)/showEvery,
                                                                r_sal_loss*(nAveGrad * self.config.batch_size)/showEvery,
                                                                r_sum_loss*(nAveGrad * self.config.batch_size)/showEvery))

                    print('Learning rate: ' + str(self.lr_bone))
                    r_edge_loss, r_sal_loss, r_sum_loss= 0,0,0

                # save obtained images at every iterations where I is a multiple of 200
                if i % 200 == 0:

                    vutils.save_image(torch.sigmoid(up_sal_f[-1].data), tmp_path+'/iter%d-sal-0.jpg' % i, normalize=True, padding = 0)

                    vutils.save_image(sal_image.data, tmp_path+'/iter%d-sal-data.jpg' % i, padding = 0)
                    vutils.save_image(sal_label.data, tmp_path+'/iter%d-sal-target.jpg' % i, padding = 0)
            
            #  save checkpoint every 'epoch_save' epochs
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net_bone.state_dict(), '%s/models/epoch_%d_bone.pth' % (self.config.save_fold, epoch + 1))

            # adjust learning rate    
            if epoch in lr_decay_epoch:
                self.lr_bone = self.lr_bone * 0.1  
                self.optimizer_bone = Adam(filter(lambda p: p.requires_grad, self.net_bone.parameters()), lr=self.lr_bone, weight_decay=p['wd'])

        # save final tained model
        torch.save(self.net_bone.state_dict(), '%s/models/final_bone.pth' % self.config.save_fold)

# binary cross-entropy loss function        
def bce2d_new(input, target, reduction=None):
    assert(input.size() == target.size())           # check for if input and target have same shape
    pos = torch.eq(target, 1).float()               # create tensor 'pos' which contains all positive elements from target
    neg = torch.eq(target, 0).float()               # create tensor 'neg' which contains all elements equal to zero

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total                    # calculates the weight factor for the negative class
    beta = 1.1 * num_pos  / num_total               # calculates the weight factor for the positive class
    weights = alpha * pos + beta * neg              # calculates the class balancing weights

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)
