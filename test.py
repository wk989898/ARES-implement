
import torch
import torch.nn.functional as F
from _test.utils import getAtoms, getAtomInfo, onehot, V_like
import os


def testEmbed():
    onehot(V[0], atoms)


# self-interaction
def testInteraction():
    from _test.model import SelfInteractionLayer
    layer = SelfInteractionLayer(3, 24)
    layer.cuda()
    out = layer(V)
    return out


# Convolution
def testConvolution():
    from _test.model import Convolution
    layer = Convolution(24, 24)
    layer.cuda()
    out = layer(V, atom_data)
    return out


# Norm
def testNorm():
    from _test.model import Norm
    layer = Norm()
    layer.cuda()
    out = layer(V)
    return out

# testNonLinearity


def testNonLinearity():
    from _test.model import NonLinearity
    layer = NonLinearity(24)
    layer.cuda()
    out = layer(V)
    return out


def testChannel():
    from _test.model import Channel_mean
    layer = Channel_mean()
    layer.cuda()
    out = layer(V)
    return out

# Denselayer


def testDense():
    from _test.model import Denselayer as Dense
    layer1 = Dense(24, 4, activation=F.elu)
    layer2 = Dense(4, 256)
    layer3 = Dense(256, 1)
    layer1.cuda()
    layer2.cuda()
    layer3.cuda()
    E.cuda()
    out = layer1(E)
    out = layer2(out)
    out = layer3(out)
    return out


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    atoms = getAtoms('3q3z.pdb')
    atom_data = getAtomInfo(atoms)
    V = V_like(len(atom_data), dim=3, cuda=True)

    testEmbed()
    V = testInteraction()
    V = testConvolution()
    V = testNorm()
    V = testNonLinearity()
    E = testChannel()
    E = testDense()

    print(E.item())
    print('test finish!')
