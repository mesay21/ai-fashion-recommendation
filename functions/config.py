import os 
import json
import numpy as numpy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torchvision
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import pad_packed_sequence
from torchvision import datasets, models, transforms

from functions.utils import seqs2batch
from functions.model import Bi_lstm
from functions.loss import LSTM_loss
from functions.data import Preprocess
from functions.data import collate_fn



batch_size = 15
save_path = '../save_path'
load_path = None
lr = 0.2
cuda = True
freeze = False
batch_first = True
dictFile = './data/final_word_dict.txt'

filenames = {'train': 'train_no_dup.json',
			 'test': 'test_no_dup.json',
			 'val': 'valid_no_dup.json'}


data_params = {'img_dir': 'data/images', 'json_dir': 'data/label', 'json_files': filenames, 'batch_size':batch_size, 'batch_first':batch_first}

optimization_params = {'learning_rate': lr, 'weight_decay': 1e-4}

def create_vocab(dictFile):
	my_vocab = {}
	file1 = open(dictFile, 'r')
	count = 0
	while True:
		count += 1
		line = file1.readline()
		if not line:
			break
		word = line.split()[0]
		idx = line.split()[1]
		my_vocab[word] = count
	count = count + 1
	my_vocab['unknown_word'] = count
	return my_vocab, count



def config(network_params, data_params, optimizer_params, cuda_params):

    vocab, vocab_size = create_vocab(dictFile)

    input_dim, hidden_dim, margin, load_path, freeze = network_params

    model = Bi_lstm(input_dim, hidden_dim, vocab_size, data_params['batch_first'],dropout=0.0, freeze=freeze)

        # 1) resize 2) crop 3) ToTensor 4) Normalize
    """
    img_transforms = {
        'train': transforms.Compose([
            transforms.Resize((305,305)),
            transforms.RandomCrop((299,299)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor()

            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((299,299)),
            #transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((299,299)),
            #transforms.CenterCrop(299),
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
    

    if load_path is not None:
        print("Loading weights from %s" % load_path)
        model.load_state_dict(torch.load(load_path))
    if cuda_params['cuda']:
        print("Switching model to gpu")
        model.cuda()

    dataloaders = {x: torch.utils.data.DataLoader(
        
        Preprocess(os.path.join(data_params['json_dir'], data_params['json_files'][x]),

                        data_params['img_dir'],
                        img_transform=img_transforms[x]),#, txt_transform=txt_transforms[x]),

        batch_size=data_params['batch_size'],
        shuffle=True, num_workers=24,
        collate_fn=collate_fn,
        pin_memory=True)
                   for x in ['train', 'test', 'val']}


    #print(dataloaders)

    # Optimize only the layers with requires_grad = True, not the frozen layers:
    optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=optimizer_params['learning_rate'], weight_decay=optimizer_params['weight_decay'])
    
    criterion = LSTM_loss(data_params['batch_first'], cuda_params['cuda'])

    return model, dataloaders, vocab, optimizer, criterion