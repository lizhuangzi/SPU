import os, sys

sys.path.append("../")
import torch
import torch.nn as nn
from torch.nn import Conv1d, Conv2d, Conv3d
from knn_cuda import KNN
from pointnet2.pointnet2_utils import gather_operation, grouping_operation
import torch.nn.functional as F
from arch.PixelShuffle1D import PixelUnshuffle1D, PixelShuffle1D
import numpy as  np


class get_edge_feature(nn.Module):
    """construct edge feature for each point
    Args:
        tensor: input a point cloud tensor,batch_size,num_dims,num_points
        k: int
    Returns:
        edge features: (batch_size,num_dims,num_points,k)
    """

    def __init__(self, k=16):
        super(get_edge_feature, self).__init__()
        self.KNN = KNN(k=k + 1, transpose_mode=False)
        self.k = k

    def forward(self, point_cloud):
        dist, idx = self.KNN(point_cloud, point_cloud)
        '''
        idx is batch_size,k,n_points
        point_cloud is batch_size,n_dims,n_points
        point_cloud_neightbors is batch_size,n_dims,k,n_points
        '''
        idx = idx[:, 1:, :]
        point_cloud_neighbors = grouping_operation(point_cloud, idx.contiguous().int())
        point_cloud_central = point_cloud.unsqueeze(2).repeat(1, 1, self.k, 1)
        # print(point_cloud_central.shape,point_cloud_neighbors.shape)
        edge_feature = torch.cat([point_cloud_central, point_cloud_neighbors - point_cloud_central], dim=1)

        return edge_feature, idx

        return dist, idx


class denseconv(nn.Module):
    def __init__(self, growth_rate=64, k=16, in_channels=6, isTrain=True):
        super(denseconv, self).__init__()
        self.edge_feature_model = get_edge_feature(k=k)
        self.growth_rate = growth_rate
        self.in_channels = in_channels
        '''
        input to conv1 is batch_size,2xn_dims,k,n_points
        '''
        self.conv1 = nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=growth_rate, kernel_size=[1, 1], padding=[0, 0]),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            Conv2d(in_channels=growth_rate + in_channels, out_channels=growth_rate, kernel_size=[1, 1], padding=[0, 0]),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            Conv2d(in_channels=2 * growth_rate + in_channels, out_channels=growth_rate, kernel_size=[1, 1],
                   padding=[0, 0]),
            nn.ReLU()
        )
        # self.conv4 = nn.Sequential(
        #     Conv2d(in_channels=3 * growth_rate + in_channels, out_channels=growth_rate*3, kernel_size=[1, 1], padding=[0, 0],groups=4),
        #     nn.ReLU()
        # )
        # self.conv5 =  nn.Sequential(
        #     Conv3d(in_channels=growth_rate,out_channels=growth_rate,kernel_size=[1,1,1],padding=[0, 0,0]),
        # )
        self.conv5 = nn.Sequential(
            Conv3d(in_channels=growth_rate,out_channels=growth_rate,kernel_size=[1,1,1],padding=[0, 0,0]),
            nn.ReLU(),
            Conv3d(in_channels=growth_rate, out_channels=growth_rate, kernel_size=[1, 1, 1], padding=[0, 0, 0]),
                                   )


    def forward(self, input):
        '''
        y should be batch_size,in_channel,k,n_points
        '''
        y, idx = self.edge_feature_model(input)
        inter_result = torch.cat([self.conv1(y), y], dim=1)  # concat on feature dimension
        preresult = torch.cat([self.conv2(inter_result), inter_result], dim=1)
        conv3out = self.conv3(preresult)

        inter_result = torch.cat([conv3out, preresult], dim=1)
        B, C, K, N = inter_result.shape
        inter_result = inter_result.view(B,self.growth_rate,int((3 * self.growth_rate + self.in_channels)/self.growth_rate),K,N)
        inter_result = self.conv5(inter_result)

        # expandresult  = self.conv4(inter_result)
        # expandresult = expandresulit.view(B,selfe.growth_rate,3,K,N)
        # expandresult = self.conv5(expandresult)
        # expandresult = torch.sum(expandresult, dim=2)
        # inter_result = torch.cat([expandresult,preresult], dim=1)
        inter_result = inter_result.view(B,C,K,N)
        final_result = torch.max(inter_result, dim=2)[0]  # pool the k channel
        return final_result, idx


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.growth_rate = 24
        self.dense_n = 3
        self.knn = 16
        self.input_channel = 3
        comp = self.growth_rate * 2
        '''
        make sure to permute the input, the feature dimension is in the second one.
        input of conv1 is batch_size,num_dims,num_points
        '''
        self.conv1 = nn.Sequential(
            Conv1d(in_channels=self.input_channel, out_channels=24, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.denseconv1 = denseconv(in_channels=24 * 2,
                                    growth_rate=self.growth_rate)  # return batch_size,(3*24+48)=120,num_points
        self.conv2 = nn.Sequential(
            Conv1d(in_channels=144, out_channels=comp, kernel_size=1),
            nn.ReLU()
        )
        self.denseconv2 = denseconv(in_channels=comp * 2, growth_rate=self.growth_rate)
        self.conv3 = nn.Sequential(
            Conv1d(in_channels=312, out_channels=comp, kernel_size=1),
            nn.ReLU()
        )
        self.denseconv3 = denseconv(in_channels=comp * 2, growth_rate=self.growth_rate)
        self.conv4 = nn.Sequential(
            Conv1d(in_channels=480, out_channels=comp, kernel_size=1),
            nn.ReLU()
        )
        self.denseconv4 = denseconv(in_channels=comp * 2, growth_rate=self.growth_rate)

    def forward(self, input):
        l0_features = self.conv1(input)  # b,24,n
        # print(l0_features.shape)
        l1_features, l1_index = self.denseconv1(l0_features)  # b,24*2+24*3=120,n
        l1_features = torch.cat([l1_features, l0_features], dim=1)  # b,120+24=144,n

        l2_features = self.conv2(l1_features)  # b,48,n
        l2_features, l2_index = self.denseconv2(l2_features)  # b,48*2+24*3=168,n
        l2_features = torch.cat([l2_features, l1_features], dim=1)  # b,168+144=312,n

        l3_features = self.conv3(l2_features)  # b,48,n
        l3_features, l3_index = self.denseconv3(l3_features)  # b,48*2+24*3=168,n
        l3_features = torch.cat([l3_features, l2_features], dim=1)  # b,168+312=480,n

        l4_features = self.conv4(l3_features)  # b,48,n
        l4_features, l4_index = self.denseconv4(l4_features)
        l4_features = torch.cat([l4_features, l3_features], dim=1)  # b,168+480=648,n

        return l4_features





class SEPS(nn.Module):
    def __init__(self,inchannel,outchannel = 256,up_ratio=4):
        super(SEPS, self).__init__()
        self.up_ratio = up_ratio
        self.conv0=nn.Sequential(
            nn.Conv1d(in_channels=inchannel,out_channels=outchannel,kernel_size=1),
            nn.ReLU()
        )
        self.conv1=nn.Sequential(
            nn.Conv1d(in_channels=inchannel,out_channels=outchannel,kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=outchannel*2,out_channels=outchannel,kernel_size=1),
            nn.ReLU()
        )

        self.convspe = Conv1d(in_channels=outchannel,out_channels=2*outchannel,kernel_size=1)
        self.convspe2 = Conv1d(in_channels=outchannel, out_channels=2 * outchannel,kernel_size=1)
        self.upshuffle = PixelShuffle1D(self.up_ratio // 2)

        # self.CA = nn.Sequential(
        #     nn.AdaptiveAvgPool1d(1),
        #     nn.Conv1d(in_channels=256,out_channels=256//16,kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(in_channels=256 // 16, out_channels=256, kernel_size=1),
        #     nn.Sigmoid()
        # )
        #self.attention = attention_unit(256)



    def forward(self,x, f2=None,stage=0):
        B, C, N = x.shape
        #x1 = x * f2
        x = self.conv0(x)
        #x2 = self.conv1(x)
        #x = torch.cat([x11,x2],dim=1)
        #x = self.conv2(x)
        #x2 = self.conv02(f2)
        x = self.convspe(x)
        x = self.upshuffle(x)
        x = self.convspe2(x)
        x = self.upshuffle(x)
        return x

class Generator(nn.Module):
    def __init__(self, params=None):
        super(Generator, self).__init__()
        self.feature_extractor = feature_extraction()

        self.upsampling = SEPS(inchannel=648)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=3, kernel_size=1)
        )

        # self.convs1 =  nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=24, kernel_size=1),
        #     nn.PReLU(),
        #     nn.Conv1d(in_channels=24, out_channels=1, kernel_size=1),
        # )

    def forward(self, input,lr_jacobian):
        lr_jacobian = F.softmin(lr_jacobian.mean(dim=1,keepdim=True), dim=2)
        #lr_jacobian = F.softmax(self.convs1(lr_jacobian), dim=1)

        features = self.feature_extractor(input)  #b,648,n # b,648,n
        #jocb = self.feature_extractor2(input*lr_jacobian)

        upf = self.upsampling(features*lr_jacobian+features) # b,128,4*n

        coord = self.conv1(upf)
        coord = self.conv2(coord)
        return coord + F.upsample_nearest(input,scale_factor=4)

    def halfforwad(self, input, previous_features, globalfeature):

        previous_features = self.feature_extractor(input)
        upf = self.upsampling(previous_features, globalfeature, stage=1)
        coord = self.conv1(upf)
        coord = self.conv2(coord)
        return coord + F.upsample_nearest(input, scale_factor=4)

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = {
        "up_ratio": 4,
        "patch_num_point": 100
    }
    generator = Generator(params).cuda()
    print('# model parameters:', sum(param.numel() for param in generator.parameters()))
    point_cloud = torch.rand(4, 3, 100).cuda()
    output, _ = generator(point_cloud)
    print(output.shape)
    # discriminator=Discriminator(params,in_channels=3).cuda()
    # dis_output=discriminator(output)
    # print(dis_output.shape)