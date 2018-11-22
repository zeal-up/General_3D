import torch 
import torch.utils.data as data 
import numpy as np 
import random 
import os,sys

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ShapenetCore_2048xyz(data.Dataset):
    def __init__(self, root='./dataset', num_points=2048, transforms=None):
        super().__init__()

        self.root = root
        self.file = os.path.join(root, 'shapenet57448xyzonly.npz')
        self.num_points = num_points
        self.transforms = transforms

        self.data = np.load(self.file)['data']

    def __getitem__(self, index):
        pc = self.data[index]
        pt_idxs = np.arange(self.num_points)
        random.shuffle(pt_idxs)
        pc = pc[pt_idxs, :]
        pc = pc_normalize(pc)

        if self.transforms is not None:
            pc = self.transforms(pc)
        else:
            pc = torch.from_numpy(pc)

        label = torch.tensor(index).type(torch.LongTensor)
        # label is to make the api consistent

        index = torch.tensor(index).type(torch.LongTensor)

        return pc, label, index

    def __len__(self):
        return self.data.shape[0] # 57448


if __name__ == '__main__':
    dataset = ShapenetCore_2048xyz(root='/home/zeal/work/data')
    print((len(dataset)))
    print(dataset[10][0].size())