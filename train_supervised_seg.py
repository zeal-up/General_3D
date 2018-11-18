import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import os

from models.Dgcnn_seg import DGCNN_segmentation

import utils.pytorch_utils as pt_utils
import data.data_utils as d_utils
import argparse

from my_utils.Plot import Visdom_Plot
from my_utils.trainer import Trainer_seg
from my_utils.BnMomentunScheduler import BnmomentumScheduler
from my_utils.ShapenetPart import ShapenetPartDataset

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
        "--num-points", type=int, default=2048,
        help="Number of points to train with. Default is 4096"
    )
    parser.add_argument(
        '--model-name', type=str, default='dgcnn',
        help='pointnet or dgcnn or dgcnn_jiehong'
    )

    parser.add_argument('--visdom-port', type=int, default=8197)
    parser.add_argument('--visdom-name', type=str, default='segmantation')

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = ShapenetPartDataset('./dataset/', phase='trainval', return_one_hot=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=4,
        pin_memory=True
        )

    test_dataset = ShapenetPartDataset('./dataset/', phase='test', return_one_hot=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=False, 
        num_workers=4, 
        pin_memory=True
        )

	# model
    if args.model_name == 'dgcnn':
        model = DGCNN_segmentation(num_parts=50, num_classes=16)
    else:
        pass

    model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.Adam(model.parameters())

    bn_lbmd = lambda epoch: max(args.bn_momentum * args.bnm_decay**(int(epoch / args.decay_step)), bnm_clip)
    lr_lbmd = lambda epoch: max(args.lr_decay**(int(epoch / args.decay_step)), lr_clip / args.lr)
    bn_scheduler = BnmomentumScheduler(model, bn_lbmd)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lbmd)

    def load_fn(batch_data):
        pc, label, one_hot = batch_data[0], batch_data[1], batch_data[2]
        data_dict = {}
        data_dict['pc'] = pc
        data_dict['one_hot_labels'] = one_hot

        return data_dict, label

    
    trainer = Trainer_seg(
        model, 
        loss_function=criterion, 
        optimizer = optimizer,
        train_loader=train_loader,
        device = device,
        viz=viz
        )

    trainer.train(
        nepochs=args.epochs,
        saved_path=os.path.join(args.saved_path, args.visdom_name),
        test_loader=test_loader,
        lr_scheduler=lr_scheduler,
        scheduler_metric=None,
        bn_scheduler=bn_scheduler,
        loader_fn=load_fn
        )