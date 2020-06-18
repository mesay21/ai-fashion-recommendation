# basics
import os
import json
import numpy as np

# torch
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import pad_packed_sequence

# functions
from functions.config import config
from functions.data import Preprocess
from functions.data import collate_seq
from functions.model import Bi_lstm 
from functions.loss import LSTM_loss
from functions.utils import seqs2batch

batch_size = 5
save_path = 'models'
load_path = None
lr = 0.2
cuda = True
freeze = False
batch_first = True
torch.manual_seed(999)

filenames = {'train': 'train_no_dup.json',
                 'test': 'test_no_dup.json',
                 'val': 'valid_no_dup.json'}



data_params = {'img_dir': 'data/images',
                   'json_dir': 'data/label',
                   'json_files': filenames,
                   'batch_size': batch_size,
                   'batch_first': batch_first}
optimizer_params = {'learning_rate': lr,
                  'weight_decay': 1e-4}



def train(train_params, dataloaders, cuda, batch_first, epoch_params):
	model, criterion, optimizer, scheduler, vocabulary, freeze = train_params
	nb_epochs, nb_save, save_path = epoch_params
	nb_iters = 0

	for epoch in range(nb_epochs):

		scheduler.step()

		for batch in dataloaders['train']:

			model.zero_grad()

			hidden = model.init_hidden(len(batch))

			images, texts, seq_lens, image_table, text_table = seqs2batch(batch, vocabulary)

			images = autograd.Variable(images)

			texts = autograd.Variable(texts)

			if cuda:
				hidden = (hidden[0].cuda(), hidden[1].cuda())
				images = images.cuda()
				texts = texts.cuda()

			# loss from (packed_real_batch, prediction), hidden = (hidden_state, cell_state)

			packed_real_batch, (image_features, text_features), (output, hidden) = model.forward(images, texts, seq_lens, image_table, text_table, hidden)

			# prepare the prediction for computing loss

			prediction_batch, _ = pad_packed_sequence(output, batch_first=batch_first) # is batch_first = False after done this? 

			# compute the loss using by non-pytorch loss function

			forward_loss, backward_loss = criterion(packed_real_batch, prediction_batch)

			# total lstm loss

			lstm_loss = forward_loss + backward_loss

			total_loss = lstm_loss 

			print('Total Loss: ', total_loss, 'Iteration Number: ', nb_iters)

			# loss.backward() computes dloss/dx for every param of x which needs requires_grad = True

			total_loss.backward()

			# opt step

			optimizer.step()

			nb_iters += 1

			if not nb_iters % nb_save:
			    if not os.path.exists(save_path):
			        os.makedirs(save_path)
			    torch.save(model.state_dict(), "%s_%d.pth" % (
			        os.path.join(save_path, 'model'), nb_iters))


# load models, dataloaders, vocab, optimizer, criterion from the config.py

model, dataloaders, vocabulary, optimizer, criterion = config(network_params=[512, 512, 0.2, load_path, freeze], data_params=data_params, optimizer_params=optimizer_params, cuda_params={'cuda': cuda})

# define the scheduler

scheduler = StepLR(optimizer, 2, 0.5)


# call train function here

train([model, criterion, optimizer, scheduler, vocabulary, freeze], dataloaders, cuda, batch_first, [100, 1000, save_path])
