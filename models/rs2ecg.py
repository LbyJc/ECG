# 仿照attention-Gated-networks/models/networks/sononet_grid_attention.py 分类和unet_grid_attention_3D.py 前半段
# 现在融合，是attention和attention融合，pooled和pooled 融合，然后再一起。
#以后可以考虑，先每个特征


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import config as CFG

from torch.nn.utils import weight_norm
############CNN#############
class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()

        self.conv1 = self._make_layer(1, 4)
        self.conv2 = self._make_layer(4, 8)
        self.conv3 = self._make_layer(8, 16)
        self.conv4 = self._make_layer(16, 32)

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class DeconvLayer(nn.Module):
    def __init__(self):
        super(DeconvLayer, self).__init__()

        self.deconv1 = self._make_layer(32, 32)
        self.deconv2 = self._make_layer(32, 16)
        self.deconv3 = self._make_layer(16, 8)
        self.deconv4 = self._make_layer(8, 4)

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3),
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
        )

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
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
class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
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
        self.conv1 = weight_norm(nn.Conv2d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp2d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv2d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp2d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
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


#############main#####################

class RS2ECG(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(RS2ECG, self).__init__()

        self.ConvLayers = ConvLayer()
        self.DeconvLayers = DeconvLayer()
        self.Transformer = Transformer(32, 3, 4, 64, 128)

        self.lp1 = nn.Linear(32 * 40, 32)
        self.lp2 = nn.Linear(3, 32)
        self.lp3 = nn.Linear(32, 4 * 640)
        self.lp4 = nn.Linear(32, 4 * 640)
        self.lp5 = nn.Linear(50 * 4 * 640, 4 * 640)

        self.tcn = TemporalConvNet(4, [256] * 9)

    def forward(self, x, posXYZ):
        batch_size = x.shape[0] #x:16，1163，50，1，640 XYZ:16,50,3
        htf = torch.zeros([batch_size, x.shape[1],x.shape[2], 32, 40]).to(CFG.device)
        for i in range(x.shape[2]):
            htf[:, :, i, :, :] = self.ConvLayers(x[:, :, i, :, :])
        # print(htf.shape)

        tf = torch.zeros([batch_size, x.shape[1],x.shape[2], 4, 640]).to(CFG.device)
        for i in range(x.shape[2]):
            tf[:, :, i, :, :] = self.DeconvLayers(htf[:, :, i, :, :])
        # print(tf.shape)

        input = self.lp1(htf.view(batch_size, x.shape[1], 50, -1))
        pos = self.lp2(posXYZ)
        input = input + pos
        # print(input.shape)

        sf = self.Transformer(input)
        sf = self.lp3(sf)
        sf = sf.view(batch_size, x.shape[1],x.shape[2], 4, 640)
        # print(sf.shape)

        cf = tf * sf
        cf = self.lp5(cf.view(batch_size,x.shape[1], -1)).view(batch_size,x.shape[1], 4, 640)
        # print(cf.shape)

        out = self.tcn(cf)
        # print(out.shape)

        return out
