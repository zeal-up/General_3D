from torch.nn.modules.module import Module
from ..functions.farthest_point_sample import _farthest_point_sample


class FarthestPointSample(Module):
    def __init__(self, num_sample_points):
        super(FarthestPointSample, self).__init__()
        self.num_sample_points = num_sample_points

    def forward(self, input):
        return _farthest_point_sample(self.num_sample_points)(input)
