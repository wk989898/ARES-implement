from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import embed, getAtomInfo, eta
from torch.nn.init import xavier_uniform_, zeros_
from cg import clebsch_gordan, O3_clebsch_gordan

class Net(nn.Module):
    '''
    20 layers
    '''

    def __init__(self, device):
        super().__init__()
        self.dim = 3
        self.model1 = Model(input_dim=3, dimension=24, device=device)
        self.model2 = Model(input_dim=24, dimension=12, device=device)
        self.model3 = Model(input_dim=12, dimension=4, device=device)
        self.channelMean = Channel_mean()
        self.dense = nn.Sequential(
            Dense(4, 4, activation=F.elu),
            Dense(4, 256),
            Dense(256, 1),
        )
        self.to(device)
        self.device = device

    def forward(self, atoms):
        '''
        For the input to the first network layer,
         we only have scalar features (angular order 𝑙 = 0) and
          a total of 𝐸 = 3 radial features
        V type:
            l:0-2
                a:atoms
                    c:0-3
                        m: 1->3->5
        E as dimension
        first 3 dimension
        '''
        # embed
        V = embed(atoms, dim=self.dim, device=self.device)

        # store atom info
        atom_data = getAtomInfo(atoms)

        V = self.model1(V, atom_data)
        V = self.model2(V, atom_data)
        V = self.model3(V, atom_data)
        E = self.channelMean(V)
        E = self.dense(E)

        return E


class Model(nn.Module):
    def __init__(self, input_dim, dimension, device) -> None:
        super().__init__()
        self.interaction1 = SelfInteractionLayer(input_dim, dimension)
        self.conv = Convolution(dimension, device=device)
        self.norm = Norm()
        self.interaction2 = SelfInteractionLayer(dimension, dimension)
        self.nl = NonLinearity(dimension)

    def forward(self, V, atom_data):
        V = self.interaction1(V)
        V = self.conv(V, atom_data)
        V = self.norm(V)
        V = self.interaction2(V)
        V = self.nl(V)
        return V


class R(nn.Module):
    '''
    distance
    '''

    def __init__(self, output_dim, hide_dim=12) -> None:
        super().__init__()
        self.dense = nn.Sequential(
            Dense(12, hide_dim, activation=F.relu),
            Dense(hide_dim, output_dim)
        )

    def forward(self, r):
        assert r.shape[1] == 12
        r = self.dense(r)
        return r


class Y(nn.Module):
    '''
    angular
    '''

    def __init__(self, dim) -> None:
        super().__init__()

        def yield_Y_fn(dim):
            if dim == 0:
                def fn(xs):
                    return torch.ones((xs.shape[0], 1)).to(xs.device)
            elif dim == 1:
                def fn(xs):
                    return xs
            elif dim == 2:
                def fn(vecs):
                    eps = 1e-9
                    r2 = torch.sum(vecs**2, dim=1).clamp_(min=eps)
                    x, y, z = vecs[..., 0], vecs[..., 1], vecs[..., 2]
                    return torch.stack([x * y / r2,
                                        y * z / r2,
                                        (-x**2 - y**2 + 2. * z**2) /
                                        (2 * 3**0.5 * r2),
                                        z * x / r2,
                                        (x**2 - y**2) / (2. * r2)],
                                       dim=-1)
            else:
                raise ValueError('angular dimension error')

            return fn

        self.Y_fn = torch.jit.script(yield_Y_fn(dim))

    def forward(self, vecs):
        return self.Y_fn(vecs)


class Convolution(nn.Module):
    def __init__(self, output_dim, device) -> None:
        super().__init__()
        self.device = device
        self.radial = R(output_dim=output_dim)
        self.angular = nn.ModuleList(
            [Y(0), Y(1), Y(2)]
        )
        # non-zero only for |𝑙𝑖 − 𝑙𝑓 | ≤ 𝑙𝑜 ≤ 𝑙𝑖 + 𝑙𝑓
        self.C = [[0, 0, 0],  [0, 1, 1],  [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 2], [0, 2, 2], [1, 2, 1], [1, 2, 2],
                  [2, 2, 0], [2, 2, 1], [2, 2, 2], [2, 0, 2], [2, 1, 1], [2, 1, 2]]
        self.CG = dict()
        for i, f, o in self.C:
            self.CG[(i, f, o)] = O3_clebsch_gordan(o, i, f, device=device)

    def forward(self, V, atom_data):
        O = defaultdict(list)
        for i, f, o in self.C:
            acif = []
            for rads, vecs, nei_idxs in atom_data:
                rads, vecs, nei_idxs = torch.tensor(
                    rads, device=self.device), torch.tensor(vecs, device=self.device), torch.tensor(nei_idxs, device=self.device)
                r = self.radial(rads)
                y = self.angular[f](vecs)
                order = torch.index_select(V[i], dim=0, index=nei_idxs)
                cif = torch.einsum('lc,lf,lci->lcif',
                                   r, y, order).sum(dim=0)
                acif.append(cif)
            assert len(acif) == V[i].shape[0]
            O[o].append(torch.einsum(
                'oif,acif->aco', self.CG[(i, f, o)], torch.stack(acif)))

        for i in range(len(O)):
            O[i] = torch.stack(O[i]).sum(dim=0)
        assert O[0].shape[-1] == 1
        assert O[1].shape[-1] == 3
        assert O[2].shape[-1] == 5
        del V

        return O


class Norm(nn.Module):
    def __init__(self, eps=1e-9) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, V):
        for key in V:
            V[key] = F.normalize(V[key], eps=self.eps)
        return V


class SelfInteractionLayer(nn.Module):
    '''
     bias term is only used when operating on angular order 0
    '''

    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.weight = nn.Parameter(xavier_uniform_(
            torch.Tensor(output_dim, input_dim)))
        self.bias = nn.Parameter(zeros_(torch.Tensor(output_dim)))

    def forward(self, V):
        O = defaultdict(list)
        for key in V:
            if key == 0:  # need bias
                O[key] = (torch.einsum('nij,ki->njk',
                                       V[key], self.weight)+self.bias).permute(0, 2, 1)
            else:
                O[key] = torch.einsum('nij,ki->nkj',
                                      V[key], self.weight)
        del V
        return O


class NonLinearity(nn.Module):
    def __init__(self, output_dim) -> None:
        super().__init__()
        self.b1 = nn.Parameter(zeros_(torch.Tensor(output_dim)))
        self.b2 = nn.Parameter(zeros_(torch.Tensor(output_dim)))
        self.bias = {
            1: self.b1,
            2: self.b2
        }
        self.output_dim = output_dim

    def forward(self, V):
        for key in V:
            if key == 0:
                V[key] = eta(V[key])
            else:
                temp = torch.sqrt(torch.einsum(
                    'acm,acm->c', V[key], V[key]))+self.bias[key]
                V[key] = torch.einsum('acm,c->acm', V[key], eta(temp))
        assert V[0].shape[-1] == 1
        assert V[1].shape[-1] == 3
        assert V[2].shape[-1] == 5
        return V


class Channel_mean(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, V):
        '''
        only V[0]
        '''
        return torch.sum(V[0], dim=0).squeeze()


def init_wb(m):
    if isinstance(m, nn.Linear):
        xavier_uniform_(m.weight)
        zeros_(m.bias)


class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, activation=None):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim).apply(init_wb)
        self.activation = activation

    def forward(self, E):
        out = self.dense(E)
        if self.activation is not None:
            out = self.activation(out)
        return out
