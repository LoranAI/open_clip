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
    
class SimplexCustom(nn.Module):
    """
    Symplex layer following the https://arxiv.org/abs/2103.15632 paper
    
    Args:
        in_features: int, input features
        out_features: int, output features
        n_classes: int, number of classes
        simplex_type: str, type of symplex to use. Options are 'd-simplex', 'd-ortoplex', 'd-cube'
    """
    def __init__(self, 
                 out_features: int, 
                 n_classes: int, 
                 simplex_type: str = 'd-symplex'):
        super().__init__()
        self.simplex_type = simplex_type
        self.out_features = out_features
        self.n_classes = n_classes
        
        if self.simplex_type == 'd-simplex':
            self.feat_size = self.n_classes - 1
            self.fixed_weight = self.d_symplex()
        elif self.simplex_type == 'd-ortoplex':
            self.feat_size = torch.ceil(torch.tensor(self.n_classes / 2)).int().item()
            self.fixed_weight = self.d_ortoplex()
        elif self.simplex_type == 'd-cube':
            self.feat_size = torch.ceil(torch.log2(torch.tensor(self.n_classes))).int().item()
            self.fixed_weight = self.d_cube()
        else:
            raise ValueError(f"simplex_type {self.simplex_type} not recognized")

        self.fc = torch.nn.Linear(self.out_features, self.feat_size, bias=False)
        self.symplex = torch.nn.Linear(self.feat_size, self.n_classes, bias=False)

        self.symplex.weight.requires_grad = False
        self.symplex.weight.copy_(self.fixed_weight)
        
    def d_symplex(self):
        """
        Symplex is the generalization of a triangle or tetrahedron to arbitrary dimensions.
        A symplex in n dimensional space is the convex hull of n+1 points that are not coplanar.
        """
        vec = torch.zeros((self.feat_size + 1, self.feat_size)) #matrix of shape (dim+1, dim)
        torch.eye(self.feat_size, out=vec[:-1,:])           
        alpha = (1.0 - torch.sqrt(1.0 + torch.tensor([self.feat_size]))) / self.feat_size
        vec[-1,:].add_(alpha) 
        vec.add_(-torch.mean(vec, dim=0)) #t = t - (1/d)
        vec.div_(torch.norm(vec, p=2, dim=1, keepdim=True)+ 1e-8)
        return vec

    def d_ortoplex(self):
        """
        The vertices of a ortoplex can be choosed as unit vector pointing algoside each coordinate 
        axis i.e. all the permutations of (-+1, 0, 0, ..., 0)
        """
        vec = torch.eye(self.feat_size)
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
        vec = torch.tensor(list(itertools.product([-1, 1], repeat=self.feat_size)), dtype=torch.float32)
        vec = vec / torch.norm(vec, p=2, dim=1, keepdim=True)
        return vec

    def forward(self, x):
        return self.symplex(self.fc(x))

"""
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
            simplex_type: str = None,
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

        if simplex_type is not None:
            self.text_projection = SimplexCustom(width, output_dim, simplex_type=simplex_type)
        else:
            if proj_bias:
                self.text_projection = nn.Linear(width, output_dim)
            else:
                self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self.init_parameters()

    def init_parameters(self):
        pass

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
    

class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
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
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
            simplex_type: str = None,
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1],\
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)

        if simplex_type is not None:
            self.proj = SimplexCustom(pool_dim, output_dim, simplex_type=simplex_type)
        else:
            self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

        self.init_parameters()


    def init_parameters(self):
        pass


    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = self.transformer(x)

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens
        
        return pooled

"""