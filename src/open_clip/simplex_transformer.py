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
    
## -----------------------------------------------------------------------------------------------
class Simplex2(nn.Module):
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
                 n_classes: int, 
                 simplex_type: str = 'd-symplex'):
        super().__init__()
        self.simplex_type = simplex_type
        self.in_features = in_features
        self.n_classes = n_classes

        # self.symplex = torch.nn.Linear(self.in_features, self.n_classes, bias=False)
        self.symplex = torch.nn.Parameter(torch.empty(self.in_features, self.n_classes))
        self.symplex.weight.requires_grad = False
        
        if self.simplex_type == 'd-simplex':
            if self.in_features != self.n_classes - 1:
                raise ValueError(f"dim must be n_classes - 1 for simplex_type {self.simplex_type}")
            self.symplex.weight.copy_(self.d_symplex())
        elif self.simplex_type == 'd-ortoplex':
            if self.in_features != torch.ceil(torch.tensor(self.n_classes/2)).int():
                self.symplex.weight.copy_(self.ortoplex())
        elif self.simplex_type == 'd-cube':
            self.target_dim = 2 ** self.in_features
            if self.target_dim != self.n_classes:
                raise ValueError(f"dim must be 2**dim for simplex_type {self.simplex_type}")
            if self.in_features != torch.ceil(torch.log2(torch.tensor(self.n_classes))).int():
                raise ValueError(f"dim must be log2(n_classes) for simplex_type {self.simplex_type}")
            self.symplex.weight.copy_(self.d_cube())
        else:
            raise ValueError(f"simplex_type {self.simplex_type} not recognized")
        
    def forward(self, x):
        return x @ self.symplex
        
    def d_symplex(self):
        """
        Symplex is the generalization of a triangle or tetrahedron to arbitrary dimensions.
        A symplex in n dimensional space is the convex hull of n+1 points that are not coplanar.
        """
        vec = torch.zeros((self.in_features + 1, self.in_features)) #matrix of shape (dim+1, dim)
        torch.eye(self.in_features, out=vec[:-1,:])           
        alpha = (1.0 - torch.sqrt(1.0 + torch.tensor([self.in_features]))) / self.in_features
        vec[-1,:].add_(alpha) 
        vec.add_(-torch.mean(vec, dim=0)) #t = t - (1/d)
        vec.div_(torch.norm(vec, p=2, dim=1, keepdim=True)+ 1e-8)
        return vec


    def dsimplex(num_classes=10, device='cuda'):
        """
        Symplex is the generalization of a triangle or tetrahedron to arbitrary dimensions.
        A symplex in n dimensional space is the convex hull of n+1 points that are not coplanar.
        """
        def simplex_coordinates(features, device):
            vec = torch.zeros((features + 1, features), device=device) # Matrix of shape (Dim+1, Dim)
            torch.eye(features, out=vec[:-1,:], device=device)
            alpha = (1.0 - torch.sqrt(1.0 + torch.tensor([features], device=device))) / features
            vec[-1,:].add_(alpha)
            vec.add_(-torch.mean(vec, dim=0)) # t = t - (1/d)
            vec.div_(torch.norm(vec, p=2, dim=1, keepdim=True) + 1e-8)
            return vec

        feat_dim = num_classes - 1
        return simplex_coordinates(feat_dim, device)
    
    
class SimplexTextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
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
        super().__init__()
        assert pool_type in ('first', 'last', 'argmax', 'none')
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.pool_type = pool_type

        self.token_embedding = nn.Embedding(vocab_size, width)
        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer('attn_mask', self.build_causal_mask(), persistent=False)

        if proj_simplex:
            self.text_projection = Simplex(width, output_dim)
        else:
            if proj_bias:
                self.text_projection = nn.Linear(width, output_dim)
            else:
                self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                nn.init.normal_(self.text_projection.weight, std=self.transformer.width ** -0.5)
                if self.text_projection.bias is not None:
                    nn.init.zeros_(self.text_projection.bias)
            else:
                nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_causal_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, _expand_token(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            if attn_mask is not None:
                attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = self.transformer(x, attn_mask=attn_mask)

        # x.shape = [batch_size, n_ctx, transformer.width]
        if self.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled, tokens = text_global_pool(x, pool_type='last')
            pooled = self.ln_final(pooled)  # final LN applied after pooling in this case
        else:
            x = self.ln_final(x)
            pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled