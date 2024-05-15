import torch
import torch.nn as nn
import itertools
from timm.models.vision_transformer import VisionTransformer


class Symplex(nn.Module):
    """
    Symplex layer following the https://arxiv.org/abs/2103.15632 paper
    
    Args:
        in_features: int, input features
        out_features: int, output features
        n_classes: int, number of classes
        symplex_type: str, type of symplex to use. Options are 'd-symplex', 'd-ortoplex', 'd-cube'
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 n_classes: int, 
                 symplex_type: str = 'd-symplex'):
        super().__init__()
        self.symplex_type = symplex_type
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes

        self.fc = torch.nn.Linear(self.in_features, self.out_features)
        self.symplex = torch.nn.Linear(self.out_features, self.n_classes, bias=False)
        self.symplex.weight.requires_grad = False
        
        if self.symplex_type == 'd-symplex':
            if self.out_features != self.n_classes - 1:
                raise ValueError(f"dim must be n_classes - 1 for symplex_type {self.symplex_type}")
            self.symplex.weight.copy_(self.d_symplex())
        elif self.symplex_type == 'd-ortoplex':
            if self.out_features != torch.ceil(torch.tensor(self.n_classes/2)).int():
                self.symplex.weight.copy_(self.ortoplex())
        elif self.symplex_type == 'd-cube':
            self.target_dim = 2 ** self.out_features
            if self.target_dim != self.n_classes:
                raise ValueError(f"dim must be 2**dim for symplex_type {self.symplex_type}")
            if self.out_features != torch.ceil(torch.log2(torch.tensor(self.n_classes))).int():
                raise ValueError(f"dim must be log2(n_classes) for symplex_type {self.symplex_type}")
            self.symplex.weight.copy_(self.d_cube())
        else:
            raise ValueError(f"symplex_type {self.symplex_type} not recognized")
        
    def forward(self, x):
        return self.symplex(self.fc(x))
        
    def d_symplex(self):
        """
        Symplex is the genrealization of a triangle or tetrahedron to arbitraty dimensions.
        A symplex in n dimensional space is the convex hull of n+1 points that are not coplanar.
        """
        vec = torch.zeros((self.out_features + 1, self.out_features)) #matrix of shape (dim+1, dim)
        torch.eye(self.out_features, out=vec[:-1,:])           
        alpha = (1.0 - torch.sqrt(1.0 + torch.tensor([self.out_features]))) / self.out_features
        vec[-1,:].add_(alpha) 
        vec.add_(-torch.mean(vec, dim=0)) #t = t - (1/d)
        vec.div_(torch.norm(vec, p=2, dim=1, keepdim=True)+ 1e-8)
        return vec

    def d_ortoplex(self, x):
        """
        The verticies of a ortoplex can be choosed as unit vector pointing algoside each coordinate 
        axis i.e. all the permutations of (-+1, 0, 0, ..., 0)
        """
        vec = torch.eye(self.out_features)
        vec = torch.cat([vec, -vec], dim=0) 
        return vec
    
    def d_cube(self):
        """
        The d-cube is the set of all binary vectors in d dimensions. 
        A unit hypercube in d-dimensional space is a convex hull of all the points whose 
        d coordinates are either 0 or 1. Can be thought as the cartesian product [0, 1]^d
        d times. The d-cube is the normalized version of the hypercube.
        """
        #vec = torch.tensor([[1 if (j >> i) % 2 == 0 else \
        #    -1 for i in range(self.out_features)] for j in range(2 ** self.out_features)])
        vec = torch.tensor(list(itertools.product([-1, 1], repeat=self.out_features)), dtype=torch.float32)
        vec = vec / torch.norm(vec, p=2, dim=1, keepdim=True)
        return vec


class VisionTransformerSymplex(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj = nn.Parameter(torch.eye(self.output_dim))