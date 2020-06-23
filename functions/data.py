import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import collections
import random
import ast

class Preprocess(Dataset):

	def __init__(self, json_file, img_dir, img_transform=None):
		self.img_dir = img_dir
		self.data = json.load(open(json_file))
		self.img_transform = img_transform


	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		set_id = self.data[index]['set_id']
		items = self.data[index]['items']
		#item_types = self.data[index]['item_types']

		images = []
		texts = []
		item_types = []
		ignored = []


		# load item type dictionary

		file = open("../data/item_type.txt", "r")
		contents = file.read()
		item_type_dictionary = ast.literal_eval(contents)
		file.close()

		#print(item_type_dictionary)

		for i in items:
			img = Image.open(os.path.join(self.img_dir, set_id, '%s.jpg' % i['index']))
			try:
				if img.layers == 1:
					img = Image.merge("RGB", [img.split()[0], img.split()[0], img.split()[0]])
			except AttributeError:
				ignored.append(set_id + '_%s' % i['index'])
				if np.any(np.array(img.size) == 1):
					continue

			images.append(img)
			texts.append(i['name'])
			if str(i['categoryid']) not in item_type_dictionary:
				item_types.append(11)
			else:
				item_types.append(int(item_type_dictionary[str(i['categoryid'])]))

		if self.img_transform:
			images = [self.img_transform(image) for image in images]
			#print(images,'data.py')

			#for image in images:
			#	print(image.size()) # 5, 4, 8, 8, 8, 4

		# shuffle 
		#test_numbers

			#c = list(zip(images, texts, item_types))
			#print(images[0],texts[0],"BEFORE SHUFFLE")

			#random.shuffle(c)

			#images, texts, item_types = zip(*c)

			#for i in range(len(images)):
			#	print(images[i],texts[i],"AFTER SHUFFLE")
		#print(item_types)
			
		return {'images': images, 'texts': texts, 'ignored': ignored, 'item_types': item_types}


def collate_seq(batch):
    if isinstance(batch[0], collections.Mapping):
        return batch

