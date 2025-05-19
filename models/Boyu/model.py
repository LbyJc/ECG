import torch, sys, os
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import resnet18

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from models.Boyu.encoder import Encoder
from models.Boyu.decoder import TemporalConvNet

from torchinfo import summary
# os.environ["CUDA_VISIBLE_DEVICES"] = "5" #sb llx


class ECGFormer(nn.Module):
    def __init__(self, encoder_parameter, num_inputs, num_channels):
        super().__init__()

        self.encoder_parameter = encoder_parameter
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        in_dim_spa_1, out_dim_spa_1, in_dim_spa_2, out_dim_spa_2, in_dim_tem, out_dim_tem, in_dim_final, out_dim_final = encoder_parameter
        self.encoder = Encoder(in_dim_spa_1, out_dim_spa_1, in_dim_spa_2, out_dim_spa_2, in_dim_tem, out_dim_tem, in_dim_final, out_dim_final)
        self.decoder = TemporalConvNet(num_inputs, num_channels, kernel_size=3)
        # self.linear_head = nn.Linear(5, 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, temporal_data, spatial_data):
        # feature_fusion: batch_sizex4x640
        feature_fusion = self.encoder(temporal_data, spatial_data)
        # feature_fusion = torch.cat((feature_fusion, history_ECG.unsqueeze(1)), 1)
        ### # feature_fusion = torch.cat((feature_fusion, history_ECG), 1)
        predicted_ECG = self.decoder(feature_fusion)
        # predicted_ECG = self.linear_head(predicted_ECG.permute(0, 2, 1))
        return predicted_ECG


if __name__ == '__main__':
    batch_size = 2
    encoder_parameter = 3, 32, [1,32], [4,640], 80, 1, 50, 1
    num_inputs = 4
    num_channels = [4]*9
    model = ECGFormer(encoder_parameter, num_inputs, num_channels)
    temporal_data = torch.randn(batch_size, 50, 1, 640)
    spatial_data = torch.randn(batch_size, 50, 3)
    history_ECG = torch.randn(batch_size, 640)

    output = model(temporal_data, spatial_data, history_ECG)
    print("output shape:", output.shape)

    print(summary(model, input_size=[(2, 50, 1, 640), (2, 50, 3), (2, 640)]))
