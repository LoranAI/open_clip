from typing import Callable, Optional
import itertools

import torch
from torch import nn
from .transformer import Transformer, VisionTransformer, TextTransformer, LayerNorm

class Simplex(nn.Module):
    """
    Symplex layer following the https://arxiv.org/abs/2103.15632 paper
    
    Args:
        in_features: int, input features
        out_features: int, output features
        n_classes: int, number of classes
        simplex_type: str, type of symplex to use. Options are 'd-simplex', 'd-ortoplex', 'd-cube'
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 n_classes: int, 
                 simplex_type: str = 'd-symplex'):
        super().__init__()
        self.simplex_type = simplex_type
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes

        self.fc = torch.nn.Linear(self.in_features, self.out_features)
        self.symplex = torch.nn.Linear(self.out_features, self.n_classes, bias=False)
        self.symplex.weight.requires_grad = False
        
        if self.simplex_type == 'd-simplex':
            if self.out_features != self.n_classes - 1:
                raise ValueError(f"dim must be n_classes - 1 for simplex_type {self.simplex_type}")
            self.symplex.weight.copy_(self.d_symplex())
        elif self.simplex_type == 'd-ortoplex':
            if self.out_features != torch.ceil(torch.tensor(self.n_classes/2)).int():
                self.symplex.weight.copy_(self.ortoplex())
        elif self.simplex_type == 'd-cube':
            self.target_dim = 2 ** self.out_features
            if self.target_dim != self.n_classes:
                raise ValueError(f"dim must be 2**dim for simplex_type {self.simplex_type}")
            if self.out_features != torch.ceil(torch.log2(torch.tensor(self.n_classes))).int():
                raise ValueError(f"dim must be log2(n_classes) for simplex_type {self.simplex_type}")
            self.symplex.weight.copy_(self.d_cube())
        else:
            raise ValueError(f"simplex_type {self.simplex_type} not recognized")
        
    def forward(self, x):
        return self.symplex(self.fc(x))
        
    def d_symplex(self):
        """
        Symplex is the generalization of a triangle or tetrahedron to arbitrary dimensions.
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
        The vertices of a ortoplex can be choosed as unit vector pointing algoside each coordinate 
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


class PolytopeTransformer(Transformer):
    """
    PolytopeTransformer is a transformer model that uses a simplex layer after the transformer

    Inherits from CLIP Transformer
    """

    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            n_classes: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            simplex_type: str = "d_simplex_type",
    ):
        """
        Initialize the PolytopeTransformer model

        :param width: int:
        :param layers: int:
        :param heads: int:
        :param n_classes: int: new attribute for the number of classes
        :param mlp_ratio: float:
        :param ls_init_value: float:
        :param act_layer: Callable:
        :param norm_layer: Callable:
        :param simplex_type: str: new attribute for the simplex type
        """

        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.n_classes = n_classes
        self.simplex_type = simplex_type
        self.out_features = None

        if self.simplex_type == "d-simplex":
            self.out_features = self.n_classes - 1
        elif self.simplex_type == "d-orthoplex":
            self.out_features = torch.ceil(torch.tensor(self.n_classes / 2)).int().item()
        elif self.simplex_type == "d-cube":
            self.out_features = torch.ceil(torch.log2(torch.tensor(self.n_classes))).int().item()
        else: 
            raise ValueError("Undefined simplex type")

        self.simplex = Simplex(self.width, self.out_features, self.n_classes, self.simplex_type)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = super().forward(x, attn_mask)

        # NOTE: check for attention mask usage.
        x = self.simplex(x)  
        return x
      

class PolytopeVisionTransformer(VisionTransformer):
    """
    PolytopeVisionTransformer is a vision transformer model that uses a simplex layer after the transformer
    
    Inherits from CLIP VisionTransformer
    """

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            n_classes: int,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            output_tokens: bool = False,
            simplex_type: str = "d_simplex_type", # Added simplex type
    ):
        """
        Initialize the PolytopeVisionTransformer model
        """

        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            attentional_pool=attentional_pool,
            attn_pooler_queries=attn_pooler_queries,
            attn_pooler_heads=attn_pooler_heads,
            output_dim=output_dim,
            patch_dropout=patch_dropout,
            no_ln_pre=no_ln_pre,
            pos_embed_type=pos_embed_type,
            pool_type=pool_type,
            final_ln_after_pool=final_ln_after_pool,
            act_layer=act_layer,
            norm_layer=norm_layer,
            output_tokens=output_tokens,
        )
        self.n_classes = n_classes
        self.simplex_type = simplex_type
        
        self.transformer = PolytopeTransformer(
            width=width,
            layers=layers,
            heads=heads,
            n_classes=n_classes,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            simplex_type=simplex_type,
        )
        
    def forward(self, x: torch.Tensor):
        return super().forward(x)
    

class PolytopeTextTransformer(TextTransformer):
    """
    PolytopeVisionTransformer is a vision transformer model that uses a simplex layer after the transformer

    Inherits from CLIP VisionTransformer
    """

    def __init__(
            self,
            n_classes: int,
            simplex_type: str = "d_simplex_type",
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            output_dim: int = 512,
            embed_cls: bool = False,
            no_causal_mask: bool = False,
            pad_id: int = 0,
            pool_type: str = 'argmax',
            proj_bias: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        """
        Initialize the PolytopeTextTransformer model

        :param n_classes: int: new attribute for the number of classes
        :param simplex_type: str: new attribute for the simplex type
        :param context_length: int:
        :param vocab_size: int:
        :param width: int:
        :param heads: int:
        :param layers: int:
        :param mlp_ratio: float:
        :param ls_init_value: float:
        :param output_dim: int:
        :param embed_cls: bool:
        :param no_causal_mask: bool:
        :param pad_id: int:
        :param pool_type: str:
        :param proj_bias: bool:
        :param act_layer: Callable:
        :param norm_layer: Callable:
        :param output_tokens: bool:
        """

        super().__init__(
            context_length=context_length,
            vocab_size=vocab_size,
            width=width,
            heads=heads,
            layers=layers,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            output_dim=output_dim,
            embed_cls=embed_cls,
            no_causal_mask=no_causal_mask,
            pad_id=pad_id,
            pool_type=pool_type,
            proj_bias=proj_bias,
            act_layer=act_layer,
            norm_layer=norm_layer,
            output_tokens=output_tokens,
        )
        self.transformer = PolytopeTransformer(
            width=width,
            layers=layers,
            heads=heads,
            n_classes=n_classes,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            simplex_type=simplex_type,
        )
        
    def forward(self, x: torch.Tensor):
        return super().forward(x)
    
