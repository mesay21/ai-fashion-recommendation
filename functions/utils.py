import torch
import PIL
import random
import numpy as np


def type_one_hot(value):
	one_hot = torch.zeros(1,12)
	one_hot[0,value] = 1
	#print(one_hot,'one_hot')
	return one_hot


def get_one_hot(text, vocabulary):
	word_embed = torch.zeros(len(text.split()), len(vocabulary)+1)

	for i, word in enumerate(text.split()):
		if word in vocabulary:
			word_embed[i, int(vocabulary[word])] = 1
		else:
			word_embed[i, int(vocabulary['unknown'])] = 1

	return word_embed




def seqs2batch(data, vocabulary):
	# inputs: 1) batch_size x seq_len (or set_length 8 images and 8 captions)
	#	   	  2) dictionary for vocabulary 
	# output: 1) image_table = [[1,2,3,4], [5,6,7,8,9], ...., [31,32,33,34,35]]
	#         2) text_table  = [[1,2,3,4,5,6,7,8], ...., []] # image caption length in items of first set is 2.  
	all_images = torch.Tensor()
	all_texts = torch.Tensor()
	all_types = torch.Tensor()

	image_data = [i['images'] for i in data] # i: a set, data: batch of sets
	text_data = [i['texts'] for i in data]
	type_data = [i['item_types'] for i in data]
	#for i in data:
#		print(i)
#	print(data)

	seq_lens = torch.zeros(len(image_data)).int()

	image_table = [None] * len(data)
	text_table = [None] * len(data)
	type_table = [None] * len(data)

	image_count = 0
	word_count = 0
	type_count = 0

	for set_index, (set_images, set_texts, set_types) in enumerate(zip(image_data, text_data, type_data)):

		image_lookup_in_set = []
		text_lookup_in_set = []
		type_lookup_in_set = []

		for image, text, types in zip(set_images, set_texts, set_types):
			text_to_append = range(word_count, word_count + len(text.split()))
			if not text_to_append:
				continue

			all_images = torch.cat((all_images, image.unsqueeze(0))) # all images in the sets of batch are combined
			#print(texts, text, 'TEXT')
			all_texts = torch.cat((all_texts, get_one_hot(text, vocabulary))) # all texts in the sets of batch are combined

			all_types = torch.cat((all_types, type_one_hot(types)))

			image_lookup_in_set.append(image_count)
			text_lookup_in_set.append(range(word_count, word_count + len(text.split())))
			type_lookup_in_set.append(type_count)

			image_count += 1 # image count increases by 1
			word_count += len(text.split()) # word count increases by words in each caption
			type_count += 1

			seq_lens[set_index] += 1

		image_table[set_index] = image_lookup_in_set
		text_table[set_index] =   text_lookup_in_set
		type_table[set_index] = type_lookup_in_set

	return all_images, all_texts, all_types, seq_lens, image_table, text_table, type_table


def predict_single_direction(ht, feats):
	scores = torch.nn.functional.log_softmax(torch.mm(ht, feats.permute(1, 0)), dim=1)
	max_score, index = torch.max(scores, 1)
	return index, torch.exp(max_score), scores


def predict_multi_direction(hf, hb, feats):
	scores = torch.nn.functional.log_softmax(torch.mm(hf, feats.permute(1, 0)), dim=1) + \
			 torch.nn.functional.log_softmax(torch.mm(hb, feats.permute(1, 0)), dim=1)
	max_score, index = torch.max(scores, 1)
	return index, torch.exp(max_score), scores









