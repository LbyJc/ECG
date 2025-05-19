############CNN#############
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import config as CFG

from torch.nn.utils import weight_norm
class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()

        self.conv1 = self._make_layers(1, 8, 1, pool=False)
        self.conv2 = self._make_layers(8, 16, 1)
        self.conv3 = self._make_layers(16, 24, 1)
        self.conv4 = self._make_layers(24, 32, 1)

    def _make_layers(self, input_features, output_features, mul, pool=True):
        mid = int(input_features * mul)
        if pool:
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
        else:
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
                , nn.BatchNorm1d(output_features))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class DeconvLayer(nn.Module):
    def __init__(self):
        super(DeconvLayer, self).__init__()

        self.conv1 = self._make_layers(32, 32)
        self.conv2 = self._make_layers(32, 16)
        self.conv3 = self._make_layers(16, 8)
        self.conv4 = self._make_layers(8, 4, deconv=False)

    def _make_layers(self, input_features, output_features, deconv=True):
        if deconv:
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
        else:
            return nn.Sequential(
                nn.Conv1d(
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
        x = self.conv1(x)
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
        """
        其实这就是一个裁剪的模块, 裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长, 一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1, 输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分, 维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  #  裁剪掉多出来的padding部分, 维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        :param num_inputs: int,  输入通道数
        :param num_channels: list, 每层的hidden_channel数, 例如[25,25,25,25]表示有4个隐层, 每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i   # 膨胀系数：1, 2, 4, 8……
            in_channels = num_inputs if i == 0 else num_channels[i-1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear_head = nn.Linear(num_channels[-1], 1)
    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        # x = x.permute(0, 2, 1)
        output = self.network(x)
        #print(output.shape) #64,256,640
        output = self.linear_head(output.permute(0, 2, 1))
        # print(output.shape) #64,640,1
        # return output.permute(0, 2, 1)
        return output
#############main#####################

class airECG_catFusion_newTCN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(airECG_catFusion_newTCN, self).__init__()

        self.ConvLayers = ConvLayer()
        self.DeconvLayers = DeconvLayer()
        self.dropout = nn.Dropout(0.5)
        self.Transformer = Transformer(32, 3, 4, 64, 128)

        self.lp1 = nn.Linear(80, 1)
        # self.lp1 = nn.Linear(32*40, 32)
        self.lp2 = nn.Linear(3, 32)
        self.lp3 = nn.Linear(32, 4 * 640)
        self.lp4 = nn.Linear(100, 1)

        self.tcn = TemporalConvNet(4, [256] * 9)
        #FCN：
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256, 250)
        #self.fc2 = nn.Linear(64, 32)
        #self.fc3 = nn.Linear(32, 10)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x, posXYZ):
        # 64,50,1,640
        htf = torch.zeros([x.shape[0], x.shape[1], 32, 80]).to(CFG.device)
        for i in range(x.shape[1]):
            htf[:, i, :, :] = self.ConvLayers(x[:, i, :, :])
        # print(htf.shape)

        tf = torch.zeros([x.shape[0], x.shape[1], 4, 640]).to(CFG.device)
        for i in range(x.shape[1]):
            tf[:, i, :, :] = self.DeconvLayers(htf[:, i, :, :])
        # print(tf.shape)

        input = self.lp1(htf).squeeze()
        # input = self.lp1(htf.view(x.shape[0],50,-1))
        pos = self.lp2(posXYZ)
        input = input + pos
        # print(input.shape)
        input = self.dropout(input)
        sf = self.Transformer(input)
        # 64,50,32
        sf = self.lp3(sf)
        sf = sf.view(x.shape[0], x.shape[1], 4, 640)
        # print(sf.shape) #64,50,4,640
        # tf:b,50,4,640
        # sf:b,50,4,640
        # cf = tf * sf
        cf = torch.cat((tf,sf),dim=1)
        cf = self.lp4(cf.permute(0, 2, 3, 1)).squeeze()
        # print(cf.shape)

        #新TCN
        output = self.tcn(cf)
        # output = self.linear_head(output.permute(0, 2, 1))
        # print(output.shape) #64,1,640 shoule be 64,256,640

        # FCNN:
        #64,256,640
        #output = self.relu(self.fc1(output.permute(0,2,1)))
        #output = self.drop(output)
        #output = self.relu(self.fc2(output))
        #output = self.fc3(output)
        #output = self.softmax(output)

        return output
