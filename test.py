import torch
import torch.nn.functional as F
from utils import getAtoms, getAtomInfo, onehot, V_like
import os


def testEmbed(V):
    onehot(V[0], atoms)


# self-interaction
def testInteraction():
    from model import SelfInteractionLayer
    layer = SelfInteractionLayer(3, 12)
    layer.cuda()
    out = layer(V)
    return out


# Convolution
def testConvolution():
    from model import Convolution
    layer = Convolution(12, 12, device='cuda')
    layer.cuda()
    out = layer(V, atom_data)
    return out


# Norm
def testNorm():
    from model import Norm
    layer = Norm()
    layer.cuda()
    out = layer(V)
    return out

# NonLinearity
def testNonLinearity():
    from model import NonLinearity
    layer = NonLinearity(12)
    layer.cuda()
    out = layer(V)
    return out

# Channel mean
def testChannel():
    from model import Channel_mean
    layer = Channel_mean()
    layer.cuda()
    out = layer(V)
    return out

# Dense
def testDense():
    from model import Dense
    dense = torch.nn.Sequential(
        Dense(12, 4, activation=F.elu),
        Dense(4, 256),
        Dense(256, 1),
    )
    dense.cuda()
    out = dense(E)
    return out

# All
def testNet():
    from model import Net
    net=Net(device='cuda')
    pred=net(atoms)
    return pred

if __name__ == '__main__':
    atoms = getAtoms('S_000028_476.pdb')
    pred=testNet()
    # atom_data = getAtomInfo(atoms)
    # V = V_like(len(atom_data), dim=3)
    # testEmbed(V)
    # V = testInteraction()
    # V = testConvolution()
    # V = testNorm()
    # V = testNonLinearity()
    # E = testChannel()
    # pred = testDense()
    print(pred.item())
    print('test finish!')
