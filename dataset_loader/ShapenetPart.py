import os
import os.path
import sys
import numpy as np
import struct
import math
import pickle

import json
import torch
import h5py
from torch.utils.data.dataloader import default_collate

def pc_normalize(pc):
	l = pc.shape[0]
	centroid = np.mean(pc, axis=0)
	pc = pc - centroid
	m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
	pc = pc / m
	return pc

def jitter_point_cloud(data, sigma=0.01, clip=0.05):
	""" Randomly jitter points. jittering is per point.
		Input:
		  BxNx3 array, original batch of point clouds
		Return:
		  BxNx3 array, jittered batch of point clouds
	"""
	N, C = data.shape
	assert(clip > 0)
	jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
	jittered_data += data
	return jittered_data
		
class ShapenetPartDataset():
	def __init__(self, data_root='./dataset/', npoint=2048, phase='train', normalize=True, return_one_hot=True, jitter=True, normal=False):
		self.phase = phase
		self.npoint = npoint
		self.normalize = normalize
		self.return_one_hot = return_one_hot
		self.jitter = jitter
		self.data_root = data_root
		self.normal = normal

		self.data_filename = os.path.join(self.data_root, 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
		self.catfile = os.path.join(self.data_filename, 'synsetoffset2category.txt')

		self.cat = {}

		with open(self.catfile, 'r') as f:
			for line in f:
				ls = line.strip().split()
				self.cat[ls[0]] = ls[1]
		self.cat = {k:v for k,v in self.cat.items()}

		self.meta = {}
		with open(os.path.join(self.data_filename, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
			train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
		with open(os.path.join(self.data_filename, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
			val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
		with open(os.path.join(self.data_filename, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
			test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

		for item in self.cat:
			self.meta[item] = []
			dir_point = os.path.join(self.data_filename, self.cat[item])
			fns = sorted(os.listdir(dir_point))
			
			if phase=='trainval':
				fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
			elif phase=='train':
				fns = [fn for fn in fns if fn[0:-4] in train_ids]
			elif phase=='val':
				fns = [fn for fn in fns if fn[0:-4] in val_ids]
			elif phase=='test':
				fns = [fn for fn in fns if fn[0:-4] in test_ids]
			else:
				print('Unknown split: %s. Exiting..'%(phase))
				exit(-1)
				
			for fn in fns:
				token = (os.path.splitext(os.path.basename(fn))[0]) 
				self.meta[item].append(os.path.join(dir_point, token + '.txt'))

		self.datapath = []
		for item in self.cat:
			for fn in self.meta[item]:
				self.datapath.append((item, fn))

		self.classes = dict(zip(self.cat, range(len(self.cat))))  
		# Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
		self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
		self.seg_label_to_classes = {} # {0:Airplane, 1:Airplane, ...49:Table}
		for class_name in self.seg_classes.keys():
			for label in self.seg_classes[class_name]:
				self.seg_label_to_classes[label] = class_name


	def __len__(self):
		return len(self.datapath)

	def __getitem__(self, index):
		fn = self.datapath[index]
		cat = self.datapath[index][0]
		cls = self.classes[cat]
		# cls = torch.IntTensor(np.array([cls]).astype(np.int32))
		one_hot_vector = torch.zeros((16, )).float()
		one_hot_vector[cls] = 1.0

		data = np.loadtxt(fn[1]).astype(np.float32)

		point_set = data[:,0:3]
		if self.normalize:
			point_set = pc_normalize(point_set)
		if self.phase=='trainval' or self.phase=='train':
			if self.jitter:
				point_set = jitter_point_cloud(point_set)
		else:
			point_set = torch.from_numpy(point_set).type(torch.FloatTensor)


		seg = data[:,-1].astype(np.int32)
		choice = np.random.choice(len(seg), self.npoint, replace=True)
		seg = torch.IntTensor(seg[choice]).long()
		point_set = torch.FloatTensor(point_set[choice, :])
		if self.normal:
			normal = data[:,3:6]
			#resample
			normal = torch.FloatTensor(normal[choice,:])
			data = torch.cat([point_set, normal], 1)
		else:
			data = point_set
		
		if self.return_one_hot:
			return data, seg, one_hot_vector
		else:
			return data, seg



if __name__ == '__main__':
	train_set = ShapenetPartDataset(phase='train')
	val_set = ShapenetPartDataset(phase='val')
	test_set = ShapenetPartDataset(phase='test')
	trainval_set = ShapenetPartDataset(phase='trainval')
	from torch.utils.data import DataLoader

	loader = DataLoader(train_set, batch_size=10, shuffle=True)
	data, seg, one_hot_vector = next(iter(loader))

	print('data.size = ', data.size())
	print('seg.size = ', seg.size())
	print('one_hot_vector.size = ', one_hot_vector.size())
	print('one_hot_vector is ', one_hot_vector)


	print('train_set length = ', len(train_set))
	print('val_set_length = ', len(val_set))
	print('trainval_length = ', len(trainval_set))
	print('test_set_length = ', len(test_set))




