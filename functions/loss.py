import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_packed_sequence


class LSTM_loss(nn.Module):
	def __init__(self, batch_first, cuda):
		super(LSTM_loss, self).__init__()
		self.batch_first = batch_first
		self.cuda = cuda

	def forward(self, truth_images_packed , hidden):
		features_batch, seq_lens_batch = pad_packed_sequence(truth_images_packed, batch_first=self.batch_first)
		forward_loss = autograd.Variable(torch.zeros(1))
		backward_loss = autograd.Variable(torch.zeros(1))

		feature_size = features_batch.size(2)
		nb_batches = len(seq_lens_batch)
		x_t_plus_1 = torch.autograd.Variable((torch.zeros(sum(seq_lens_batch)+2*nb_batches, feature_size)))

		if self.cuda:
			forward_loss = forward_loss.cuda()
			backward_loss = backward_loss.cuda()

		if self.cuda: 
			x_t_plus_1 = x_t_plus_1.cuda()


		start = 0

		# populate the x_t_plus_1
		for feature_per_seq, len_per_seq in zip(features_batch, seq_lens_batch):
			x_t_plus_1[(start + 1): (start + 1) + len_per_seq] = feature_per_seq[:len_per_seq]
			start = start + (len_per_seq + 2)


		acc_seq_lens_batch = [0]
		acc_seq_lens_batch.extend([int(k) for k in torch.cumsum(torch.FloatTensor(seq_lens_batch.float()),0)])


		for i, seq_len in enumerate(seq_lens_batch):

			# separate the ht[:first_part], ht[first_part:]
			forward_seq_hidden = hidden[i, :seq_len,  :hidden.size()[2] // 2]
			backward_seq_hidden = hidden[i, :seq_len, hidden.size()[2] // 2:]

			# Forward Loss 
			forward_log_prob = torch.nn.functional.log_softmax(torch.mm(forward_seq_hidden, x_t_plus_1.permute(1,0)), dim=1)
			seq_index_start = 2*i + acc_seq_lens_batch[i]

			forward_index_start = seq_index_start + 2
			forward_log_prob_square = forward_log_prob[:, forward_index_start:forward_index_start+forward_log_prob.size(0)]
			forward_loss = forward_loss - torch.diag(forward_log_prob_square).mean()

			# Backward Loss 
			backward_log_prob = torch.nn.functional.log_softmax(torch.mm(backward_seq_hidden, x_t_plus_1.permute(1,0)), dim=1)
			backward_index_start = seq_index_start # updated with forward_index_start already
			backward_log_prob_square = backward_log_prob[:, backward_index_start:backward_index_start+forward_log_prob.size(0)]

			backward_loss = backward_loss - torch.diag(backward_log_prob_square).mean()

		return forward_loss/len(seq_lens_batch), backward_loss/len(seq_lens_batch)

