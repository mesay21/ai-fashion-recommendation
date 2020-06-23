"""Evaluate results with a trained model."""
import os
import sys
import numpy as np
from PIL import Image
import torch
from loss import LSTM_loss
#from utils import ImageTransforms
import torchvision
from sklearn import metrics
from torchvision import transforms

class Evaluation(object):
    """Evaluate an existing model.

    Args:
        model (pytorch model)
        weights (str): path to the saved weights.

    """
    def __init__(self, model, weights, img_dir, batch_first, cuda, batch_size=20):
        """Load the model weights."""
        if cuda:
            self.model = model.cuda()
        else:
            self.model = model
        self.model.eval()
        self.model.load_state_dict(torch.load(weights))
        self.img_dir = img_dir
        #self.model_type = model_type
        """
        IMG_TRF = transforms.Compose([
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.trf = IMG_TRF#lambda x: IMG_TRF.resize(x)
        """
        """
        img_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop((299,299)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                #transforms.Resize(256),
                transforms.CenterCrop((299,299)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                #transforms.Resize(256), 
                transforms.CenterCrop((299,299)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        """
        img_transforms = {
            'train': transforms.Compose([
                transforms.Resize((240)),
                transforms.RandomCrop((224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((240)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(240),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        #self.trf = img_transforms['test']
        
        self.criterion = LSTM_loss(batch_first, cuda=cuda)
        self.batch_first = batch_first
        self.cuda = cuda



    def get_images(self, sequence):
        """Get a list of images from a list of names."""
        images = []
        for im_path in sequence:
            img = Image.open(os.path.join(self.img_dir, im_path.replace('_', '/') + '.jpg'))
            try:
                if img.layers == 1:  # Imgs with 1 channel are usually noise.
                    # continue
                    img = Image.merge("RGB", [img.split()[0], img.split()[0], img.split()[0]])
            except AttributeError:
                # Images with size == 1 in any dimension are useless.
                if np.any(np.array(img.size) == 1):
                    continue
            images.append(img)

        return images

    def get_img_feats(self, img_data):
        images = torch.Tensor()
        for img in img_data:
            #images = torch.cat((images, transforms.ToTensor()(self.trf(img)).unsqueeze(0)))
            #images = torch.cat((images, transfroms.RandomResizedCrop(299)..unsqueeze(0)))
            #TRF = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            #IMG_TRF = image_transforms['test']
            #self.trf = lambda x: TRF(torchvision.transforms.ToTensor()(IMG_TRF.resize(x)))

            img = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(240),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])(img)
            images = torch.cat((images, img.unsqueeze(0)))
        images = torch.autograd.Variable(images)
        if self.cuda:
            images = images.cuda()
        return self.model.cnn(images)



#model_path = '../models/model_331000.pth'
#feats_path = 'save_path'
