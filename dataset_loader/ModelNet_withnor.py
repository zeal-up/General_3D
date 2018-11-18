import torch
import torch.utils.data as data
import numpy as np 
import random
import os, h5py


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ModelNet40_10_withnor():
    def __init__(self, root, num_points=1024, train=True, normalize=True, normals=True, modelnet10=False):
        self.root = os.path.join(root, 'modelnet40_normal_resampled')
        self.num_points = num_points
        self.normalize = normalize
        self.train = train

        if modelnet10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        self.normals = normals
        
        if train:
            if modelnet10:
                shape_ids= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))] 
            else:
                shape_ids = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))] 
        else:
            if modelnet10:
                shape_ids = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
            else: 
                shape_ids = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[i])+'.txt') for i in range(len(shape_ids))]
        self.num_classes = len(self.cat)

    def __getitem__(self, index): 
        fn = self.datapath[index]
        label = self.classes[fn[0]]
        point_set = np.loadtxt(fn[1],delimiter=',').astype(np.float32)
        pt_idx = np.arange(self.num_points)
        random.shuffle(pt_idx)
        point_set = point_set[pt_idx, :]
        if self.normalize:
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.normals:
            point_set = point_set[:, 0:3]
        point_set = torch.from_numpy(point_set)
        label = torch.tensor(label).type(torch.LongTensor)
        index = torch.tensor(index).type(torch.LongTensor)
        return point_set, label, index
        
    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    train_set = ModelNet40_10_withnor(root='/home/zeal/work/data')
    test_set = ModelNet40_10_withnor(root='/home/zeal/work/data', train=False)

    print(len(train_set))
    print(len(test_set))

    print(train_set[10][0].size())