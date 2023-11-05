import torch
import torch.nn as nn
import torch.nn.functional as F


class DihedralFeatures(nn.Module):
    def __init__(self, node_embed_dim):
        """ Embed dihedral angle features. """
        super(DihedralFeatures, self).__init__()
        # 3 dihedral angles; sin and cos of each angle
        node_in = 6
        # Normalization and embedding
        self.node_embedding = nn.Linear(node_in,  node_embed_dim, bias=True)
        self.norm_nodes = Normalize(node_embed_dim)

    def forward(self, X):
        """ Featurize coordinates as an attributed graph """
        with torch.no_grad():
            V = self._dihedrals(X)
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        return V

    @staticmethod
    def _dihedrals(X, eps=1e-7, return_angles=False):
        # First 3 coordinates are [N, CA, C] / [C4', C1', N1/N9]
        if len(X.shape) == 4:
            X = X[..., :3, :].reshape(X.shape[0], 3*X.shape[1], 3)
        else:
            X = X[:, :3, :]

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))

        # phi, psi, omega = torch.unbind(D,-1)
        #
        # if return_angles:
        #     return phi, psi, omega

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features

class DihedralFeatures_atom7(nn.Module):
    """ Embed dihedral angle features. Using 7 atoms"""

    def __init__(self, node_embed_dim):
        """ Embed dihedral angle features. """
        super(DihedralFeatures_atom7, self).__init__()
        # 4 dihedral angles; sin and cos of each angle
        node_in = 6 + 6
        # Normalization and embedding
        self.node_embedding = nn.Linear(node_in, node_embed_dim, bias=True)
        self.norm_nodes = Normalize(node_embed_dim)

    def forward(self, X):
        """ Featurize coordinates as an attributed graph """
        with torch.no_grad():
            V = self.rna_dihedrals(X)
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        return V

    @staticmethod
    def _cal(cord_tns):
        eps = 1e-6
        x1, x2, x3, x4 = [torch.squeeze(x, dim=2) for x in torch.split(cord_tns, 1, dim=2)]
        a1 = x2 - x1
        a2 = x3 - x2
        a3 = x4 - x3
        v1 = torch.cross(a1, a2, dim=2)
        v1 = v1 / (torch.norm(v1, dim=2, keepdim=True) + eps)
        v2 = torch.cross(a2, a3, dim=2)
        v2 = v2 / (torch.norm(v2, dim=2, keepdim=True) + eps)
        sign = torch.sign(torch.sum(v1 * a3, dim=2))
        sign[sign == 0.0] = 1.0  # to avoid multiplication with zero
        rad_vec = sign * torch.arccos(torch.clip(
            torch.sum(v1 * v2, dim=2) / (torch.norm(v1, dim=2) * torch.norm(v2, dim=2) + eps), -1.0, 1.0))

        return rad_vec

    @staticmethod
    def backbone_dihedrals(X, eps=1e-7):
        # First 3 coordinates are [N, CA, C] / [C4', C1', N1/N9]
        if len(X.shape) == 4:
            X = X[..., :3,:].reshape(X.shape[0], 3*X.shape[1], 3)
        else:
            X = X[:, :3, :]

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features

    def rna_dihedrals(self, X):
        n1 = self._cal(X[..., [0, 1, 2, 3], :]).unsqueeze(-1)  # C4' C1' N1 C2 -> B * L * 1
        n2 = self._cal(X[..., [2, 1, 0, 4], :]).unsqueeze(-1)  # N1 C1' C4' C5'
        n4 = self._cal(X[..., [0, 4, 5, 6], :]).unsqueeze(-1)  # C4' C5' O5' P

        D = torch.cat((n1, n2, n4), dim=-1)  # B*L*3
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)

        return torch.cat([D_features, self.backbone_dihedrals(X)], dim=-1)


class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias