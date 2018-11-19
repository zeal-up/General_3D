import torch
import torch.nn as nn
from torch.autograd import Variable
from query_ball_point.modules.query_ball_point import QueryBallPoint

def _get_distance(queries, points):
	r_a = torch.sum(queries * queries, dim=1, keepdim=True)
	r_b = torch.sum(points * points, dim=1, keepdim=True)
	dis = torch.bmm(torch.transpose(queries, 1, 2), points).mul_(-2)
	dis.add_(r_a.transpose(1, 2) + r_b)

	return dis

xyz1 = torch.rand(2, 3, 50)*2-1
xyz1 = Variable(xyz1.float().cuda())
xyz2 = torch.index_select(xyz1, 2, Variable(torch.arange(10).long().cuda()))
dis = _get_distance(xyz2, xyz1)
mask = dis<=(0.4**2)

net = QueryBallPoint(0.4, 10).cuda()
idx, pts_idx = net(xyz1, xyz2)
print(idx[0][0])
print(torch.nonzero(mask[0][0])[0])
