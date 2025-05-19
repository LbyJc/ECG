import torch, os, sys
# from data.spectrum_dataset import SpectrumECGDataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch.nn as nn
import torch.nn.functional as F
# from nets.units import *
import math
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

class CNNDownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, int((in_channels+out_channels)/2), kernel_size=(1, 7), padding=(0, 3))
        self.conv_2 = nn.Conv2d(int((in_channels+out_channels)/2), out_channels, kernel_size=(1, 7), padding=(0, 3))
        self.batchnorm_1 = nn.BatchNorm2d(int((in_channels+out_channels)/2))
        self.batchnorm_2= nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()


    def forward(self, x):
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.batchnorm_1(x)
        x = self.conv_2(x)
        x = self.activation(x)
        x = self.batchnorm_2(x)
        return x
    
class CNNUpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride_size):
        super().__init__()
        self.transposeConv = nn.ConvTranspose2d(in_channels, in_channels, (1, stride_size), stride=(1,stride_size))
        self.conv_1 = nn.Conv2d(in_channels, int((in_channels+out_channels)/2), kernel_size=(1, 7), padding=(0, 3))
        self.conv_2 = nn.Conv2d(int((in_channels+out_channels)/2), out_channels, kernel_size=(1, 7), padding=(0, 3))


    def forward(self, x):
        x = self.transposeConv(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x
    
    
class CNNDownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1d_1 = CNNDownSampleBlock(1, 4)
        self.cnn1d_2 = CNNDownSampleBlock(4, 8)
        self.cnn1d_3 = CNNDownSampleBlock(8, 16)
        self.cnn1d_4 = CNNDownSampleBlock(16, 32)
        self.maxpooling = nn.MaxPool2d((1, 2), stride=(1, 2))


    def forward(self, x):
        x = self.cnn1d_1(x)
        x = self.maxpooling(x)
        x = self.cnn1d_2(x)
        x = self.maxpooling(x)
        x = self.cnn1d_3(x)
        x = self.maxpooling(x)
        x = self.cnn1d_4(x)
        return x
    
    
class CNNUpSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1d_1 = CNNUpSampleBlock(32, 16, 2)
        self.cnn1d_2 = CNNUpSampleBlock(16, 16, 1)
        self.cnn1d_3 = CNNUpSampleBlock(16, 8, 2)
        self.cnn1d_4 = CNNUpSampleBlock(8, 4, 2)

    def forward(self, x):
        x = self.cnn1d_1(x)
        x = self.cnn1d_2(x)
        x = self.cnn1d_3(x)
        x = self.cnn1d_4(x)
        return x
    
class Positional_Embedding(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_emd = nn.Parameter(torch.randn(seq_len, dim))

    def forward(self, x):
        return x + self.pos_emd
    
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, dropout, forward_expansion): 
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion*embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query):
        attention, _ = self.attention(value, key, query)
        x = self.norm1(query + self.dropout(attention))
        # x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.norm1(x + self.dropout(forward))
        # out = self.dropout(self.norm2(forward + x))
        return out
    
class Encoder(nn.Module):
    def __init__(self, in_dim_spa_1, out_dim_spa_1, in_dim_spa_2, out_dim_spa_2, in_dim_tem, out_dim_tem, in_dim_final, out_dim_final): 
        super().__init__()
        '''
        initialize network
        Parameters:
        in_dim_spa_1/out_dim_spa_1: first linear projection in/out dimension for spatial feature reshape
        in_dim_spa_2/out_dim_spa_2: second linear projection in/out dimension for spatial feature reshape
        in_dim_tem/out_dim_tem: linear projection in/out dimension for temporal feature reshape
        in_dim_final/out_dim_final: linear projection in/out dimension for final fused feature reshape
        '''
        self.out_dim_spa_1 = out_dim_spa_1
        self.out_dim_spa_2 = out_dim_spa_2
        self.out_dim_final = out_dim_final
        self.downSample = CNNDownSample()
        self.upSample = CNNUpSample()
        self.positional_embedding = Positional_Embedding(50, 32)
        self.linear_spatial_1 = nn.Linear(in_dim_spa_1, out_dim_spa_1)
        # self.linear_spatial_2 = nn.Linear(in_dim_spa_2[0]*in_dim_spa_2[1], out_dim_spa_2[0]*out_dim_spa_2[1])
        self.linear_spatial_1_4 = nn.Linear(1,  4)
        self.linear_spatial_32_640 = nn.Linear(32, 640)
        self.linear_temporal = nn.Linear(in_dim_tem, out_dim_tem)
        self.linear_final = nn.Linear(in_dim_final,  out_dim_final)
        self.transformer = TransformerBlock(32, 4, 0.001, 128)
        self.linear_temp_only = nn.Linear(50, 1)
        
        
    def forward(self, temporal_data, spatial_data):
        # temporal data down sample by 1D CNN
        # 50 x 1 x 640 -> 50 x 32 x 80
        temporal_data = temporal_data.permute(0, 2, 1, 3)
        temporal_feature = self.downSample(temporal_data)
        temporal_feature = temporal_feature.permute(0, 2, 1, 3)
        
        # spatical data linear projection
        # 50 x 3 -> 50 x 32
        # pos_embedding = self.linear_spatial_1(spatial_data)
        # pos_embedding = self.Position_Embedding(spatial_data)
        
        # temporal data linear projection
        # 50 x 32 x 80 -> 50 x 32
        tem_embedding = self.linear_temporal(temporal_feature).squeeze(3)

        # concatenate temporal and spatial data
        # two 50 x 32 -> 50 x 32
        input = self.positional_embedding(tem_embedding)
        # input = tem_embedding + pos_embedding
        # no position embedding
        # input = tem_embedding
        
        # multi-head attention by Transformer blocks
        # 50 x 1 x 32 -> 50 x 1 x 32
        spatial_feature = self.transformer(input, input, input)
        # spatial_feature = input
        
        # spatial data linear projection
        # 50 x 1 x 32 -> 50 x 4 x 640
        shape = spatial_feature.shape
        # spa_flatten = torch.flatten(spatial_feature, 2)
        # spatial_feature = self.linear_spatial_2(spa_flatten)
        # spatial_feature = spatial_feature.reshape(shape[0], shape[1], self.out_dim_spa_2[0], self.out_dim_spa_2[1])
        
        # add by zyy, with output size 50x4x640
        spatial_feature = self.linear_spatial_1_4(spatial_feature.unsqueeze(2).permute(0,1,3,2)).permute(0,1,3,2)
        spatial_feature = self.linear_spatial_32_640(spatial_feature)


        # temporal feature up sample by 1D CNN
        # 50 x 32 x 80 -> 50 x 4 x 640
        temporal_feature = temporal_feature.permute(0, 2, 1, 3)
        temporal_feature = self.upSample(temporal_feature)
        temporal_feature = temporal_feature.permute(0, 2, 1, 3)
        
        # ﬁnal cardiac features are fused by dot producting between temporal feature and spatial feature
        # 50 x 4 x 640 -> 50 x 4 x 640
        # final_feature = torch.matmul(temporal_feature, spatial_feature.permute(0, 1, 3, 2))
        # add by zyy, with output size 50x4x640
        # final_feature = spatial_feature * temporal_feature 
        final_feature = temporal_feature 
        
        # final feature linear projection
        # 50 x 4 x 640 -> 4 x 640
        shape = final_feature.shape
        #### final_feature = final_feature.permute(0, 3, 1, 2)
        #### final_flatten = torch.flatten(final_feature, 1)
        final_feature = self.linear_final(final_feature.permute(0, 2, 3, 1)).squeeze(3)
        #### final_feature = final_feature.reshape(shape[0], shape[2], self.out_dim_final//shape[2])


        '''
        # test for only using temporal feature
        # 50 x 4 x 640 -> 4 x 640
        final_feature = temporal_feature.permute(0, 3, 2, 1)
        # final_flatten = torch.flatten(final_feature, 1)
        final_feature = self.linear_temp_only(final_feature).squeeze(3)
        final_feature = final_feature.permute(0, 2, 1)
        '''
        
        # test for only using spatial feature
        
        return final_feature


if __name__ == '__main__':
    batch_size = 2
    encoder = Encoder(3, 32, [1,32], [4,640], 80, 1, 50, 1)
    temporal_data = torch.randn(batch_size, 50, 1, 640)
    spatial_data = torch.randn(batch_size, 50, 3)
    output = encoder(temporal_data, spatial_data)
    print(output.shape)
