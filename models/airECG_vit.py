############CNN#############
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
import config as CFG

from torch.nn.utils import weight_norm
class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()

        self.conv1 = self._make_layers(1, 16, 4)
        self.conv2 = self._make_layers(16, 128, 4)
        self.conv3 = self._make_layers(128, 32, 0.5)

    def _make_layers(self, input_features, output_features, mul):
        mid = int(input_features * mul)
        return nn.Sequential(
            nn.Conv1d(
                in_channels=input_features
                , out_channels=mid
                , kernel_size=7
                , padding=3
            )
            , nn.ReLU()
            , nn.BatchNorm1d(mid)
            , nn.Conv1d(
                in_channels=mid
                , out_channels=output_features
                , kernel_size=7
                , padding=3
            )
            , nn.ReLU()
            , nn.BatchNorm1d(output_features)
            , nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DeconvLayer(nn.Module):
    def __init__(self):
        super(DeconvLayer, self).__init__()

        self.conv2 = self._make_layers(32, 16)
        self.conv3 = self._make_layers(16, 8)
        self.conv4 = self._make_layers(8, 4)

    def _make_layers(self, input_features, output_features):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=input_features
                , out_channels=input_features
                , kernel_size=2
                , stride=2
            )
            , nn.Conv1d(
                in_channels=input_features
                , out_channels=output_features
                , kernel_size=7
                , padding=3
            )
            , nn.Conv1d(
                in_channels=output_features
                , out_channels=output_features
                , kernel_size=7
                , padding=3
            )
        )

    def forward(self, x):
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


############transformer##############
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


##########TCN################
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
        # return x[:, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
            self
            , n_inputs, n_outputs, kernel_size
            , stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(
            self
            , num_inputs
            , num_channels
            , dilation_factor=2
            , kernel_size=2
            , dropout=0.2
    ):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = dilation_factor ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(
                in_channels, out_channels
                , kernel_size
                , stride=1
                , dilation=dilation_size
                , padding=(kernel_size - 1) * dilation_size
                , dropout=dropout
            )]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)


    def forward(self, series,posXYZ):
        # x = self.to_patch_embedding(series)
        x = series
        b, n = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        x +=posXYZ
        x, ps = pack([cls_tokens, x], 'b * d')

        #x += self.pos_embedding[:, :(n + 1)]

        # x +=posXYZ
        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return cls_tokens


#############main#####################

class airECG_vit(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(airECG_vit, self).__init__()

        self.ConvLayers = ConvLayer()
        self.DeconvLayers = DeconvLayer()
        self.dropout = nn.Dropout(0.5)
        self.Transformer = ViT(seq_len=32,patch_size=8, dim=32, depth =3, heads = 4, dim_head = 64, mlp_dim =128)

        self.lp1 = nn.Linear(80, 1)
        # self.lp1 = nn.Linear(32*40, 32)
        self.lp2 = nn.Linear(3, 32)
        self.lp3 = nn.Linear(32, 4 * 640)
        self.lp4 = nn.Linear(100, 1)

        self.tcn = TemporalConvNet(4, [256] * 9)

    def forward(self, x, posXYZ):
        # x:64,50,1,640 pos:64,50,3
        htf = torch.zeros([x.shape[0], x.shape[1], 32, 80]).to(CFG.device)
        for i in range(x.shape[1]):
            htf[:, i, :, :] = self.ConvLayers(x[:, i, :, :])
        # print(htf.shape) 64,50,32,80

        tf = torch.zeros([x.shape[0], x.shape[1], 4, 640]).to(CFG.device)
        for i in range(x.shape[1]):
            tf[:, i, :, :] = self.DeconvLayers(htf[:, i, :, :])
        # print(tf.shape) 64,50,4,640

        input = self.lp1(htf).squeeze()
        pos = self.lp2(posXYZ)
        # input = input + pos
        # sf = self.Transformer(input)

        #::change start::
        # input:64,50,32
        sf = torch.zeros([x.shape[0], x.shape[1], 32]).to(CFG.device)
        for i in range(x.shape[1]):
            sf[:, i, :]= self.Transformer(input[:, i, :],pos[:, i, :])
        # sf = self.Transformer(input,pos)
        #::change end::
        #64,51,32
        sf = self.lp3(sf)
        sf = sf.view(x.shape[0], x.shape[1], 4, 640)
        # print(sf.shape)

        cf = torch.cat((tf, sf), dim=1)
        cf = self.lp4(cf.permute(0, 2, 3, 1)).squeeze()
        # print(cf.shape)

        out = self.tcn(cf)
        # print(out.shape)

        return out
