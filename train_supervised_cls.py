import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import os

from models.DGCNN.Dgcnn_cls import DGCNN_cls_fullnet
from models.SpiderCNN.SpiderCNN_cls import Spidercnn_cls_fullnet
from models.PointNet2.Pointnet2_cls import Pointnet2MSG_cls_fullnet
import utils.pytorch_utils as pt_utils
import utils.data_utils as d_utils
import argparse

from utils.Plot import Visdom_Plot
from utils.BnMomentunScheduler import BnmomentumScheduler
from utils.Trainer import Trainer_cls

from dataset_loader.ModelNet40_h5py import ModelNet40_h5
from dataset_loader.ModelNet_withnor import ModelNet40_10_withnor

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size. Default is 32"
        )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="L2 regularization coeff. Default is 1e-4"
    )
    parser.add_argument(
        "--epochs", type=int, default=250, 
        help="Number of epochs to train for. Default is 200"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate. Default is 1e-3"
    )
    parser.add_argument(
        "--lr-decay", type=float, default=0.7,
        help="Learning rate decay gamma. Default is 0.7"
    )
    parser.add_argument(
        "--decay-step", type=float, default=20,
        help="Learning rate decay step.(/epoch). Default is 20"
    )
    parser.add_argument(
        "--pre-trained", type=str, default=None,
        help="pre-trained model path"
    )
    parser.add_argument(
        '--saved-path', type=str, default='./supervised_models/', 
        help='path to save model'
    )
    parser.add_argument(
        "--bn-momentum", type=float, default=0.5,
        help="Initial batch norm momentum. Default is 0.5"
    )
    parser.add_argument(
        "--bnm-decay", type=float, default=0.5,
        help="Batch norm momentum decay gamma. Default is 0.5"
    )
    parser.add_argument(
        "--num-points", type=int, default=1024,
        help="Number of points to train with. Default is 1024"
    )
    parser.add_argument(
        '--model-name', type=str, default='dgcnn',
        help='pointnet2 or dgcnn or spidercnn'
    )
    parser.add_argument(
        '--withnor', action='store_true', default=False,
        help='whether to use normals'
    )
    parser.add_argument(
        '--optim', type=str, default='adam',
        help='what kind of optimizer to use, adam, sgd'
    )

    parser.add_argument('--visdom-port', type=int, default=8197)
    parser.add_argument('--visdom-name', type=str, default='supervised')

    return parser.parse_args()


lr_clip = 1e-5
bnm_clip = 1e-2
if __name__ == "__main__":
    args = parse_args()
    viz = Visdom_Plot(title=args.visdom_name, port=args.visdom_port, env_name=args.visdom_name)
    viz.append_text(str(vars(args)))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    np.random.seed(10)

    transforms_train = transforms.Compose([
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudRotate(),
        d_utils.PointcloudJitter(),
        d_utils.PointcloudScale(),
        d_utils.PointcloudRotatePerturbation(),
        d_utils.PointcloudTranslate()
    ])

    transforms_test = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    if args.withnor:
        train_set = ModelNet40_10_withnor(root='./dataset', transforms=transforms_train, num_points=args.num_points, train=True)
        test_set = ModelNet40_10_withnor(root='./dataset', transforms=transforms_test, num_points=args.num_points, train=False)
    else:
        train_set = ModelNet40_h5(root='./dataset', transforms=transforms_train, num_points=args.num_points, train=True)
        test_set = ModelNet40_h5(root='./dataset', transforms=transforms_test, num_points=args.num_points, train=False)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True
    )

    num_classes = train_set.num_classes

    if args.model_name == 'dgcnn':
        model = DGCNN_cls_fullnet(num_classes=num_classes)
    elif args.model_name == 'spidercnn':
        model = Spidercnn_cls_fullnet(withnor=True, num_classes=num_classes)
    elif args.model_name == 'pointnet2':
        model = Pointnet2MSG_cls_fullnet(num_classes=num_classes)
    else:
        assert False, 'illegal model name'
    model = nn.DataParallel(model)
    if args.pre_trained is not None:
        model.load_state_dict(torch.load(args.pre_trained))

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    bn_lbmd = lambda epoch: max(args.bn_momentum * args.bnm_decay**(int(epoch / args.decay_step)), bnm_clip)
    lr_lbmd = lambda epoch: max(args.lr_decay**(int(epoch / args.decay_step)), lr_clip / args.lr)
    bn_scheduler = BnmomentumScheduler(model, bn_lbmd)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lbmd)
    loss_fn = F.cross_entropy
    
    trainer = Trainer_cls(
        model, 
        loss_function=loss_fn, 
        optimizer = optimizer,
        train_loader=train_loader,
        device = device,
        viz=viz
        )

    trainer.train(
        nepochs = args.epochs,
        saved_path=os.path.join(args.saved_path, args.visdom_name),
        test_loader=test_loader,
        lr_scheduler=lr_scheduler,
        scheduler_metric=None,
        bn_scheduler=bn_scheduler,
        loader_fn=None
        )