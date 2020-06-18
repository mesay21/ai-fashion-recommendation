import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision.models as models

from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models.inception import model_urls

class Bi_lstm(nn.Module):
	def __init__(self, input_size, hidden_size, vocab_size, batch_first=False, dropout=0.0, freeze=False):
		super(Bi_lstm, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.batch_first = batch_first
		self.vocab_size = vocab_size
		self.text_linear_layer = nn.Linear(vocab_size, input_size)
		self.cnn = models.resnet18(pretrained=True)
		self.num_ftrs = self.cnn.fc.in_features


		if freeze:
			for param in self.cnn.parameters():
				param.requires_grad = False

		self.cnn.fc = nn.Linear(self.num_ftrs, input_size)
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=self.batch_first, bidirectional=True, dropout=dropout)


	def forward(self, images, texts, seq_lens, image_table, text_table, hidden):
		image_features = self.cnn(images)
		image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
		word_features = self.text_linear_layer(texts)

		text_features_matrix = autograd.Variable(torch.zeros(len(images), word_features.size()[1]))
		if image_features[0].is_cuda:
			text_features_matrix = text_features_matrix.cuda()
		table_indices = [y for x in text_table for y in x]

		for i in range(text_features_matrix.size(0)):
			text_features_matrix[i, ] = torch.mean(word_features[table_indices[i]], 0)

		text_features_matrix = torch.nn.functional.normalize(text_features_matrix, p=2, dim=1)

		packed_batch_features = self.prepare_packed_seq(image_features, seq_lens, image_table)

		return packed_batch_features, (image_features, text_features_matrix), self.lstm(packed_batch_features, hidden)


	def image_forward(self, images, seq_lens, image_table, hidden):
		image_features, _ = self.cnn(images)
		packed_batch_features = self.prepare_packed_seq(image_features, seq_lens, image_table)
		return self.lstm(packed_batch_features, hidden)


	def init_hidden(self, batch_size):
		hidden_state = autograd.Variable(torch.rand(2, batch_size, self.hidden_size) * 2 * 0.08)
		cell_state = autograd.Variable(torch.rand(2, batch_size, self.hidden_size) * 2 * 0.08)
		return (hidden_state, cell_state)


	def prepare_packed_seq(self, features, seq_lens, image_table):
		seqs_tensors = autograd.Variable(torch.zeros((len(seq_lens), max(seq_lens), features.size()[1]))).cuda()

		for i, seq_len in enumerate(seq_lens):
			for j in range(max(seq_lens)):
				if j < seq_len:
					seqs_tensors[i, j] = features[image_table[i][j]] # parse through the images -- 32 images x 512 dimension
				else:
					seqs_tensors[i, j] = autograd.Variable(torch.zeros(features.size()[1])) #### just add torch.zeros(1x512) to complete the remaining sequences

		#print(seqs_tensors.size(), 'sequence size')



		seqs_tensors = seqs_tensors[sorted(range(len(seq_lens)), key=lambda k: seq_lens[k], reverse=True), :]

		batch_sorted_seq_lens = sorted(seq_lens, reverse=True)


		if not self.batch_first:
			seqs_tensors = seqs_tensors.permute(1, 0, 2) 

		return pack_padded_sequence(seqs_tensors, batch_sorted_seq_lens, batch_first=self.batch_first)
