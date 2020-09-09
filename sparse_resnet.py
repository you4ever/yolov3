import torch
import torch.nn as nn
import sparseconvnet as scn

class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential(
            scn.SubmanifoldConvolution(2, 3, 8, 3, False),
            scn.MaxPooling(2, 3, 2),
            scn.SparseResNet(2, 8, [
                        ['b', 8, 2, 1],
                        ['b', 16, 2, 2],
                        ['b', 24, 2, 2],
                        ['b', 32, 2, 2]]),
            scn.Convolution(2, 32, 64, 5, 1, False),
            scn.BatchNormReLU(64),
            scn.SparseToDense(2, 64))
        self.spatial_size= self.sparseModel.input_spatial_size(torch.LongTensor([1, 1]))
        self.inputLayer = scn.InputLayer(2,self.spatial_size,2)
        self.linear = nn.Linear(64, 183)

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.sparseModel(x)
        x = x.view(-1, 64)
        x = self.linear(x)
        return x
