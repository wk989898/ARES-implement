import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from _test.utils import V_like, getAtomInfo, radial_fn, onehot, eta, eps
from torch.nn.init import xavier_uniform_, zeros_

from cg import clebsch_gordan

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class R(nn.Module):

    def __init__(self, output_dim, hide_dim=12) -> None:
        super().__init__()
        self.dense = nn.Sequential(
            Denselayer(12, hide_dim, activation=F.relu),
            Denselayer(hide_dim, output_dim),

            # Dense(12, hide_dim, activation=F.relu),
            # Dense(hide_dim, output_dim)
        )

    def forward(self, distance):
        R = radial_fn(distance)
        R = torch.tensor(R).to(device)
        assert R.shape[0] == 12
        R = self.dense(R)
        return R


class Y(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, vec):
        vec = list(vec)
        r2 = torch.tensor(max(sum([a**2 for a in vec]), eps))
        if self.dim == 2:
            x, y, z = vec
            return torch.stack([x * y / r2,
                                y * z / r2,
                                (-x**2 - y**2 + 2. * z**2) /
                                (2 * math.sqrt(3) * r2),
                                z * x / r2,
                                (x**2 - y**2) / (2. * r2)],
                               dim=-1).to(device)
        if self.dim == 1:
            return torch.tensor(vec).to(device)
        if self.dim == 0:
            return torch.ones((1)).to(device)


class Convolution(nn.Module):

    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.C = [[0, 0, 0],  [0, 1, 1],  [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 2], [0, 2, 2], [1, 2, 1], [1, 2, 2],
                  [2, 2, 0], [2, 2, 1], [2, 2, 2], [2, 0, 2], [2, 1, 1], [2, 1, 2]]

    def forward_init(self, V, atoms):
        '''
        non-zero only for |𝑙𝑖 − 𝑙𝑓 | ≤ 𝑙𝑜 ≤ 𝑙𝑖 + 𝑙𝑓
        '''

        return self.forward(V, atoms)

    @torch.jit.script
    def forward(self, V: dict, atom_data: list):
        O = V_like(len(atom_data), self.output_dim, cuda=True)
        for i, f, o in self.C:
            acif = []
            for info in atom_data:
                cif = 0
                radial = R(output_dim=self.output_dim).to(device)
                angular = Y(dim=f).to(device)
                for mod, vec, nei_idx in info:

                    r = radial(mod)
                    y = angular(vec)
                    cif = cif + torch.einsum('c,f,ci->cif',
                                             r, y, V[i][nei_idx])
                acif.append(cif)

            assert len(acif) == V[i].shape[0]

            cg = clebsch_gordan(o, i, f).to(device)
            O[o].add_(torch.einsum(
                'oif,acif->aco', cg, torch.stack(acif)))
        assert O[0].shape[-1] == 1
        assert O[1].shape[-1] == 3
        assert O[2].shape[-1] == 5
        return O


class Norm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, V):
        '''
        doesn't need struct O
        '''
        for key in V:
            V[key] = F.normalize(V[key], eps=eps)
        return V


class SelfInteractionLayer(nn.Module):
    '''
    SchNet
     bias term is only used when operating on angular order 0
    '''

    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.weight = xavier_uniform_(
            torch.Tensor(output_dim, input_dim)).to(device)
        self.bias = zeros_(torch.Tensor(output_dim)).to(device)

    def forward(self, V):
        '''
        maybe [3,1]->[24,1]
        judge new struct O
        '''
        if self.output_dim != V[0].shape[1]:
            O = V_like(V[0].shape[0], dim=self.output_dim, cuda=True)
        else:
            O = V
        assert O[0].shape[1] == self.output_dim

        for key in V:
            if key == 0:
                O[key] = (torch.einsum('nij,ki->njk',
                                       V[key], self.weight)+self.bias).permute(0, 2, 1)
            else:
                O[key] = torch.einsum('nij,ki->nkj',
                                      V[key], self.weight)
        return O


class NonLinearity(nn.Module):

    def __init__(self, output_dim) -> None:
        super().__init__()
        self.bias = {
            1: zeros_(torch.Tensor(output_dim)).to(device),
            2: zeros_(torch.Tensor(output_dim)).to(device)
        }
        self.output_dim = output_dim

    def forward_before(self, V):

        self.forward(V)

    def forward(self, V):
        '''
        O
        '''
        for key in V:
            if key == 0:
                V[key] = eta(V[key])
            else:
                temp = torch.sqrt(torch.einsum(
                    'acm,acm->c', V[key], V[key]))+self.bias[key]
                V[key] = torch.einsum('acm,c->acm', V[key], temp)
        assert V[0].shape[-1] == 1
        assert V[1].shape[-1] == 3
        assert V[2].shape[-1] == 5
        return V


class Channel_mean(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, V):
        # print('channel',torch.sum(V[0],dim=0))
        return torch.sum(V[0], dim=0).squeeze()


class Dense(torch.nn.Linear):
    def __init__(self, input_dim, output_dim, bias=True, activation=None,
                 weight_init=xavier_uniform_, bias_init=zeros_):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.activation = activation
        super(Dense, self).__init__(input_dim, output_dim, bias)

    def reset_parameters(self):
        """
        Reinitialize model parameters.
        """
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, E):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Output of the dense layer.
        """
        y = super(Dense, self).forward(E)
        if self.activation:
            y = self.activation(y)
        return y


def init_wb(m):
    if isinstance(m, nn.Linear):
        xavier_uniform_(m.weight)
        zeros_(m.bias)


class Denselayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation=False):
        super().__init__()
        self.input_dim = input_dim
        self.activation = activation
        self.linear = nn.Linear(input_dim, output_dim,
                                bias=True).apply(init_wb)

    def forward(self, E):
        assert E.shape == torch.Size([self.input_dim])
        E = self.linear(E)
        if self.activation:
            E = self.activation(E, inplace=True)
        return E
