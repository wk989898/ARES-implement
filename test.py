import torch
import torch.nn.functional as F
from utils import help, embed


def testEmbed(dim=3):
    V = embed(atoms, dim)
    return V


def testInteraction():
    from model import SelfInteractionLayer
    layer = SelfInteractionLayer(3, 12)
    layer.cuda()
    out = layer(V)
    return out


def testConvolution():
    from model import Convolution
    layer = Convolution(12)
    layer.cuda()
    out = layer(V, atom_data)
    return out


def testNorm():
    from model import Norm
    layer = Norm()
    layer.cuda()
    out = layer(V)
    return out


def testNonLinearity():
    from model import NonLinearity
    layer = NonLinearity(12)
    layer.cuda()
    out = layer(V)
    return out


def testChannel():
    from model import Channel_mean
    layer = Channel_mean()
    layer.cuda()
    out = layer(V)
    return out


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


def testNet():
    from model import Net
    net = Net(device='cuda')
    pred = net(atoms, atom_data)
    return pred


if __name__ == '__main__':
    atoms, atom_info, rms = help('S_000041_026.pdb')
    atom_data = [torch.tensor(info).to('cuda') for info in atom_info]
    pred = testNet()
    # V = testEmbed()
    # V = testInteraction()
    # V = testConvolution()
    # V = testNorm()
    # V = testNonLinearity()
    # E = testChannel()
    # pred = testDense()
    print(pred.item(), rms)
    print('test finish!')
