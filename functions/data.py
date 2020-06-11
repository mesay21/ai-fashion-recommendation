import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import collections


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
		images = []
		texts = []
		ignored = []
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

		if self.img_transform:
			images = [self.img_transform(image) for image in images]
			
		return {'images': images, 'texts': texts, 'ignored': ignored}


def collate_seq(batch):
    if isinstance(batch[0], collections.Mapping):
        return batch

