import torch
import torch.nn as nn
import torch.nn.functional as F


'''
image_size: 이미지의 크기
patch_size: 이미지를 패치로 나눌 크기
in_channels: 입력 이미지의 채널 수
num_classes: 분류해야 하는 클래스 수
embed_dim: 패치 임베딩의 차원
depth: 인코더 블록의 수
num_heads: 멀티 헤드 어텐션의 헤드 수
mlp_ratio: MLP 모듈에서 첫 번째 FC 레이어와 두 번째 FC 레이어의 차원 비율 (기본값은 4.0)
qkv_bias: 어텐션의 행렬 연산에서 Q, K, V 행렬에 대한 바이어스 사용 여부 (기본값은 False)
attn_drop: 어텐션 드롭아웃 비율 (기본값은 0.0)
proj_drop: 어텐션 이후 프로젝션 드롭아웃 비율 (기본값은 0.0)
'''


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, mlp_drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(mlp_drop)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = (q @ k.transpose(-2, -1)) * self.scale
        x = x.softmax(dim=-1)
        x = self.attn_drop(x)
        x = (x @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0,
                 qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads,
                              qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       out_features=dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MLPHead(nn.Module):
    def __init__(self, embed_dim, mlp_hidden_dim, num_classes):
        super(MLPHead, self).__init__()
        self.embed_dim = embed_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(embed_dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.fc3 = nn.Linear(mlp_hidden_dim, num_classes)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, num_classes, embed_dim, 
                 depth, num_heads, mlp_ratio=4.0, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size=image_size, patch_size=patch_size,
                                          in_channels=in_channels, embed_dim=embed_dim)
        
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=proj_drop)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
            
            for _ in range(depth)
        ])
        self.mlp_head = MLPHead(embed_dim=embed_dim, mlp_hidden_dim=embed_dim * 4,
                                num_classes=num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        x = x[:, 0]
        x = self.mlp_head(x)
        return x