import torch
import PIL
import random
import numpy as np

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
	images = torch.Tensor()
	texts = torch.Tensor()

	image_data = [i['images'] for i in data] # i: a set, data: batch of sets
	text_data = [i['texts'] for i in data]
	seq_lens = torch.zeros(len(image_data)).int()

	image_table = [None] * len(data)
	text_table = [None] * len(data)

	image_count = 0
	word_count = 0

	for set_index, (set_images, set_texts) in enumerate(zip(image_data, text_data)):

		image_lookup_in_set = []
		text_lookup_in_set = []

		for image, text in zip(set_images, set_texts):
			text_to_append = range(word_count, word_count + len(text.split()))
			if not text_to_append:
				continue

			images = torch.cat((images, image.unsqueeze(0))) # all images in the sets of batch are combined
			#print(texts, text, 'TEXT')
			texts = torch.cat((texts, get_one_hot(text, vocabulary))) # all texts in the sets of batch are combined

			image_lookup_in_set.append(image_count)
			text_lookup_in_set.append(range(word_count, word_count + len(text.split())))

			image_count += 1 # image count increases by 1
			word_count += len(text.split()) # word count increases by words in each caption

			seq_lens[set_index] =+ 1

		image_table[set_index] = image_lookup_in_set
		text_table[set_index] =   text_lookup_in_set

	return images, texts, seq_lens, image_table, text_table







