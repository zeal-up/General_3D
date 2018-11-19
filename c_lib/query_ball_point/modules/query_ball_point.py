from torch.nn.modules.module import Module
from ..functions.query_ball_point import _query_ball_point


class QueryBallPoint(Module):
    def __init__(self, radius, nsample):
        super(QueryBallPoint, self).__init__()
        self.radius = radius
        self.nsample = nsample

    def forward(self, xyz1, xyz2):
        return _query_ball_point(self.radius, self.nsample)(xyz1, xyz2)
