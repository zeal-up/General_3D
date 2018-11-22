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

def _get_data_filename(listfile):
    with open(listfile) as f:
        return [line.strip()[5:] for line in f]

class ModelNet40_h5(data.Dataset):

    def __init__(self, root, num_points, transforms=None, train=True, random_sample=False):
        '''
        输出：
        __getitems__ : points, label, idx
        '''
        super().__init__()
        self.transforms = transforms
        self.root = root
        self.folder = "modelnet40_ply_hdf5_2048"
        # url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
        self.data_dir = os.path.join(root, self.folder)
        self.num_points = num_points
        self.random_sample = random_sample
        if train:
            self.files = \
                _get_data_filename(os.path.join(self.data_dir, 'train_files.txt'))
        else:
            self.files = \
                _get_data_filename(os.path.join(self.data_dir, 'test_files.txt'))
        
        point_list, label_list = [], []

        for f_name in self.files:
            f = h5py.File(os.path.join(self.root, f_name))
            points = f['data'][:]
            # print(points.shape)
            labels = f['label'][:]
            point_list.append(points)
            label_list.append(labels)
        
        self.points = np.concatenate(point_list, 0)
        self.labels = torch.from_numpy(np.concatenate(label_list, 0)).type(torch.LongTensor).view(-1,)
        self.num_classes =  self.labels.max().item() + 1



    def __getitem__(self, idx):

        current_points = self.points[idx].copy()
        if self.random_sample:
            pt_idxs = np.random.choice(current_points.shape[0], self.num_points, replace=False)
            random.shuffle(pt_idxs)
        else:
            pt_idxs = np.arange(self.num_points)
            random.shuffle(pt_idxs)
        pc = current_points[pt_idxs, :]
        pc = pc_normalize(pc)
        # print(pc.shape)
        if self.transforms is not None:
            pc = self.transforms(pc)
        label = self.labels[idx]
        label = torch.tensor(label).type(torch.LongTensor)
        index = torch.tensor(idx).type(torch.LongTensor)

        return pc, label, index

    def __len__(self):
        return self.points.shape[0]
    

if __name__ == '__main__':
    from torchvision import transforms
    import os,sys
    sys.path.append(os.path.abspath('.'))
    
    import utils.data_utils as d_utils 

    transforms = transforms.Compose([
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudRotate(axis=np.array([1,0,0])),
        d_utils.PointcloudScale(),
        d_utils.PointcloudTranslate(),
        d_utils.PointcloudJitter()
    ])
    train_set = ModelNet40_h5(root = "/home/zeal/work/data/", num_points=1024, transforms=transforms, train=True)
    test_set = ModelNet40_h5(root = "/home/zeal/work/data/", num_points=1024, transforms=transforms, train=False)
    print(len(train_set))
    print(len(test_set))

    print(train_set[10][0].size())