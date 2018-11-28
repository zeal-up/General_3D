# General_3D
A project to implement some state-of-the-art 3D network architecture in PyTorch.
My research topics is Geometric learning, so I need to use some state-of-the art 3D network architecture in my work.
However, most of the code provided by the paper authors is implemented by TensorFlow.
I am more familiar with PyTorch, so I want to reproduce the paper results in PyTorch.

## Existing network in this repo.

+ Have been finished and tested

    + PointNet++ : 90.76% classification accuracy on ModelNet40. 
    + DGCNN : 91.4% classification accuracy on ModelNet40(still not the result in the original paper). 85.1% mIoUs on ShapeNetpart.

+ Have not finished.

    + PointCNN : I can only get 87% accuracy on ModelNet40 and I cannot find out what's wrong in my code. I really need some help.
    + SpiderNet : notfinished.

## Dataset

+ [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)
+ [ShapeNetPart with normals](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)

After download the dataset, you have to make a directory `dataset` in the root directory, and move the above dataset to that directory.

## compile cuda kernel

To run the pointnet++ network, you have to first compile the `Farthest sampling` and `query ball point` module.

```
cd c_lib
sh make.sh
```
If you account for some problems in this stages, raise a issue.

## Dependency

+ PyTorch (above 4.0)
+ python 3.6
+ [visdom ](https://github.com/facebookresearch/visdom)
visdom can visulize the training process and it's easy to use.

## Training and evalution

Pointnet++ can only run in GPU and others should be able to run without GPU. It you run in a pulbic server, make sure to add `CUDA_VISIBLE_DEVICES=1,2...`

+ train pointnet++

```
CUDA_VISIBLE_DEVICES=1 python train_supervised_cls.py --model-name 'pointnet' --visdom-name 'pointnet_cls'
```
for other network, you only have to replace the `--model-name` which can be `pointnet`, `dgcnn`, `pointcnn`, `spidercnn`.

## Need some helps

I creat this repo to share my code and ask for some help. I can not reproduce `PointCNN` and cannot find out what's wrong in my code. So, if you are interested in this work, help me to test the `PointCNN` architecture.

## Feel free to raise a issue to ask questions