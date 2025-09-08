from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
class Bottleneck(nn.Module):
    # Standard bottleneck
    # ���ʼ�C to C���м��ѹ����resnet
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    # ������n*C to C��ѹ����������Ĺ���
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        return x


class MLP(nn.Module):
    def __init__(self, dim,N):
        super().__init__()

        self.C3=C3(dim,dim,N)

    def forward(self, x):
        # x = self.norm(x)
        x=self.C3(x)

        return x


class SCAGIA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim // 2, dim // 2, 3, padding=1, groups=dim // 2)
        self.qkvl = nn.Conv2d(dim // 2, (dim // 4) * 4, 1, padding=0)
        self.pool_q = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.pool_k = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.act = nn.GELU()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1_1, x1_2 = torch.split(x1, [C // 2, C // 2], dim=1)
        x2_1, x2_2 = torch.split(x2, [C // 2, C // 2], dim=1)

        x1_1 = self.act(self.dwconv(x1_1))
        x2_1 = self.act(self.dwconv(x2_1))
        x1_1_1,x1_1_2=torch.split(x1_1,[C // 4, C // 4], dim=1)
        x2_1_1,x2_1_2=torch.split(x2_1,[C // 4, C // 4], dim=1)
        x1_1=torch.cat([x1_1_1,x2_1_2],dim=1)
        x2_1=torch.cat([x2_1_1,x1_1_2],dim=1)


        x1_2 = self.act(self.qkvl(x1_2))
        x1_2 = x1_2.reshape(B, 4, C // 4, H, W)
        q1 = x1_2[:, 0, :, :, :]
        k1 = x1_2[:, 1, :, :, :]
        q1 = self.pool_q(q1)
        k1 = self.pool_k(k1)
        v1_1 = x1_2[:, 2, :, :, :].flatten(2)
        v1_2 = x1_2[:, 3, :, :, :].flatten(2)

        x2_2 = self.act(self.qkvl(x2_2))
        x2_2 = x2_2.reshape(B, 4, C // 4, H, W)
        q2 = x2_2[:, 0, :, :, :]
        k2 = x2_2[:, 1, :, :, :]
        q2 = self.pool_q(q2)
        k2 = self.pool_k(k2)
        v2_1 = x2_2[:, 2, :, :, :].flatten(2)
        v2_2 = x2_2[:, 3, :, :, :].flatten(2)

        # qk->[b,c,c]
        q1k1 = torch.matmul(q1.flatten(2), k1.flatten(2).transpose(1, 2))
        q1k1 = torch.softmax(q1k1, dim=1).transpose(1, 2)

        q1k2 = torch.matmul(q1.flatten(2), k2.flatten(2).transpose(1, 2))
        q1k2 = torch.softmax(q1k2, dim=1).transpose(1, 2)

        q2k2 = torch.matmul(q2.flatten(2), k2.flatten(2).transpose(1, 2))
        q2k2 = torch.softmax(q2k2, dim=1).transpose(1, 2)

        q2k1 = torch.matmul(q2.flatten(2), k1.flatten(2).transpose(1, 2))
        q2k1 = torch.softmax(q2k1, dim=1).transpose(1, 2)

        x1_s = torch.matmul(q1k1, v1_1).reshape(B, C // 4, H, W)
        x1_c = torch.matmul(q1k2, v1_2).reshape(B, C // 4, H, W)

        x2_s = torch.matmul(q2k2, v2_1).reshape(B, C // 4, H, W)
        x2_c = torch.matmul(q2k1, v2_2).reshape(B, C // 4, H, W)

        x1 = torch.cat([x1_1, x1_s, x1_c], dim=1)
        x2 = torch.cat([x2_1, x2_s, x2_c], dim=1)

        return x1, x2


class EncoderBlock(nn.Module):
    def __init__(self, dim, N):
        super().__init__()

        self.layer_norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.mlp = MLP(dim,N)
        self.attn = SCAGIA(dim)

    def forward(self, x1, x2):
        inp_copy1 = x1
        inp_copy2 = x2

        x1 = self.layer_norm1(x1)
        x2 = self.layer_norm1(x2)

        x1,x2 = self.attn(x1,x2)

        out1 = x1 + inp_copy1
        out2 = x2 + inp_copy2

        x1 = self.layer_norm2(out1)
        x1 = self.mlp(x1)
        x1 = out1 + x1

        x2 = self.layer_norm2(out2)
        x2 = self.mlp(x2)
        x2 = out2 + x2

        return x1,x2



class Encoder(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[64, 128, 256, 512], depths=[3, 3, 4, 3],N_C=1):
        super().__init__()
        self.depths = depths
        self.embed_dims = embed_dims

        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        self.block1 = nn.ModuleList()
        for i in range(depths[0]):
            self.block1.append(EncoderBlock(dim=embed_dims[0], N=N_C))

        self.block2 = nn.ModuleList()
        for i in range(depths[1]):
            self.block2.append(EncoderBlock(dim=embed_dims[1], N=N_C))

        self.block3 = nn.ModuleList()
        for i in range(depths[2]):
            self.block3.append(EncoderBlock(dim=embed_dims[2], N=N_C))

        self.block4 = nn.ModuleList()
        for i in range(depths[3]):
            self.block4.append(EncoderBlock(dim=embed_dims[3], N=N_C))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        outs1 = []
        outs2 = []
        # stage 1
        x1= self.patch_embed1(x1)
        x2= self.patch_embed1(x2)
        for blk in self.block1:
            x1, x2 = blk(x1, x2)
        outs1.append(x1)
        outs2.append(x2)

        # stage 2
        x1 = self.patch_embed2(x1)
        x2 = self.patch_embed2(x2)
        for blk in self.block2:
            x1, x2 = blk(x1, x2)
        outs1.append(x1)
        outs2.append(x2)

        # stage 3
        x1 = self.patch_embed3(x1)
        x2 = self.patch_embed3(x2)
        for blk in self.block3:
            x1, x2 = blk(x1, x2)
        outs1.append(x1)
        outs2.append(x2)

        # stage 4
        x1 = self.patch_embed4(x1)
        x2 = self.patch_embed4(x2)
        for blk in self.block4:
            x1, x2 = blk(x1, x2)
        outs1.append(x1)
        outs2.append(x2)

        return outs1,outs2

class Fusion_Block(nn.Module):
    def __init__(self, in_channels):
        super(Fusion_Block, self).__init__()
        self.proj1 = nn.Conv2d(in_channels*2, in_channels, 1, stride=1, padding=0, bias=True)
        self.proj2 = nn.Conv2d(in_channels*2, in_channels, 1, stride=1, padding=0, bias=True)
        self.bn1=nn.BatchNorm2d(in_channels)
        self.bn2=nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()

    def forward(self, x1,x2):
        concat = torch.cat([x1,x2], dim=1)
        diff=torch.abs(x1-x2)
        x = self.proj1(concat)
        x = self.act(self.bn1(x))
        x = torch.cat([x,diff],dim=1)
        x = self.proj2(x)
        x = self.act(self.bn2(x))

        return x

class Refinement_module(nn.Module):
    def __init__(self, in_channels):
        super(Refinement_module, self).__init__()
        self.us=nn.Upsample(scale_factor=2)
        self.conv1_1=nn.Conv2d(in_channels*2,in_channels,1,1)
        self.conv2_1=nn.Conv2d(in_channels,in_channels,1,1)
        self.conv3_1=nn.Conv2d(in_channels,in_channels,1,1)
        self.conv1_3=nn.Conv2d(in_channels*2,1,3,1,padding=1)

    def forward(self, x, y):
        B,C,H,W=x.shape
        y1=self.us(self.conv1_1(y))
        x1=self.conv2_1(x)
        atten=self.conv1_3(torch.cat([y1,x1],dim=1))
        atten=atten.flatten(2)
        atten=torch.softmax(atten,dim=2).reshape(B, 1, H, W)
        x=self.conv3_1(x)
        x=x*atten

        return x
    
class UpsampleConvLayer(torch.nn.Module):
    """
    Transpose convolution layer to upsample the feature maps
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class Dense_Decode(nn.Module):
    def __init__(self, in_channels,num_anchors,num_classes,N,N1,US_N):
        super(Dense_Decode, self).__init__()


        

        
        
        self.P2downsample1=nn.Sequential(nn.Conv2d(in_channels//8,in_channels//4,kernel_size=2,stride=2,padding=0),nn.BatchNorm2d(in_channels//4),nn.GELU())
        self.P2downsample2=nn.Sequential(nn.Conv2d(in_channels//8,in_channels//2,kernel_size=4,stride=4,padding=0),nn.BatchNorm2d(in_channels//2),nn.GELU())
        self.P2downsample3=nn.Sequential(nn.Conv2d(in_channels//8,in_channels,kernel_size=8,stride=8,padding=0),nn.BatchNorm2d(in_channels),nn.GELU())

        self.P3us1=nn.Sequential(nn.Conv2d(in_channels//4,in_channels//8,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(in_channels//8),nn.GELU())
        self.P3downsample1=nn.Sequential(nn.Conv2d(in_channels//4,in_channels//2,kernel_size=2,stride=2,padding=0),nn.BatchNorm2d(in_channels//2),nn.GELU())
        self.P3downsample2=nn.Sequential(nn.Conv2d(in_channels//4,in_channels,kernel_size=4,stride=4,padding=0),nn.BatchNorm2d(in_channels),nn.GELU())

        self.P4downsample1=nn.Sequential(nn.Conv2d(in_channels//2,in_channels,kernel_size=2,stride=2,padding=0),nn.BatchNorm2d(in_channels),nn.GELU())
        self.P4us1 = nn.Sequential(nn.Conv2d(in_channels//2,in_channels//4,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(in_channels//4),nn.GELU())
        self.P4us2 = nn.Sequential(nn.Conv2d(in_channels//2,in_channels//8,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(in_channels//8),nn.GELU())

        self.P5us1 = nn.Sequential(nn.Conv2d(in_channels,in_channels//2,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(in_channels//2),nn.GELU())
        self.P5us2 = nn.Sequential(nn.Conv2d(in_channels,in_channels//4,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(in_channels//4),nn.GELU())
        self.P5us3 = nn.Sequential(nn.Conv2d(in_channels,in_channels//8,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(in_channels//8),nn.GELU())


        # P4_
        self.stage1_1=nn.Sequential(nn.Conv2d(in_channels//2*4,in_channels//2*3,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(in_channels//2*3),nn.GELU())
        self.stage1=C3(in_channels//2*3,in_channels//2,N)

        self.P4_downsample1=nn.Sequential(nn.Conv2d(in_channels//2,in_channels,kernel_size=2,stride=2,padding=0),nn.BatchNorm2d(in_channels),nn.GELU())
        self.P4_us1=nn.Sequential(nn.Conv2d(in_channels//2,in_channels//4,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(in_channels//4),nn.GELU())
        self.P4_us2=nn.Sequential(nn.Conv2d(in_channels//2,in_channels//8,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(in_channels//8),nn.GELU())


        # P3_
        self.stage2_1=nn.Sequential(nn.Conv2d(in_channels//4*5,in_channels//4*3,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(in_channels//4*3),nn.GELU())
        self.stage2=C3(in_channels//4 * 3,in_channels//4,N)

        self.P3_downsample1=nn.Sequential(nn.Conv2d(in_channels//4,in_channels//2,kernel_size=2,stride=2,padding=0),nn.BatchNorm2d(in_channels//2),nn.GELU())
        self.P3_downsample2=nn.Sequential(nn.Conv2d(in_channels//4,in_channels,kernel_size=4,stride=4,padding=0),nn.BatchNorm2d(in_channels),nn.GELU())
        self.P3_us1=nn.Sequential(nn.Conv2d(in_channels//4,in_channels//8,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(in_channels//8),nn.GELU())

        # P2_
        self.stage3_1 = nn.Sequential(
            nn.Conv2d(in_channels // 4 * 3, in_channels // 8 * 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels // 8 * 3), nn.GELU())
        self.stage3 = C3(in_channels // 8 * 3, in_channels // 8, N)

        self.P2_downsample1 = nn.Sequential(
            nn.Conv2d(in_channels // 8, in_channels // 4, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(in_channels // 4), nn.GELU())
        self.P2_downsample2 = nn.Sequential(
            nn.Conv2d(in_channels // 8, in_channels//2, kernel_size=4, stride=4, padding=0), nn.BatchNorm2d(in_channels//2),
            nn.GELU())
        self.P2_downsample3 = nn.Sequential(
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=8, stride=8, padding=0), nn.BatchNorm2d(in_channels),
            nn.GELU())

        # P3__
        self.stage4_1 = nn.Sequential(
            nn.Conv2d(in_channels // 4 * 7, in_channels // 4 * 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels // 4 * 3), nn.GELU())
        self.stage4 = C3(in_channels // 4 * 3, in_channels // 4, N)

        self.P3__downsample1 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(in_channels // 2), nn.GELU())
        self.P3__downsample2 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels , kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(in_channels ),
            nn.GELU())

        # P4__
        self.stage5_1=nn.Sequential(nn.Conv2d(in_channels//2*8,in_channels//2*3,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(in_channels//2*3),nn.GELU())
        self.stage5=C3(in_channels//2 * 3,in_channels//2,N)

        self.P4__downsample1=nn.Sequential(nn.Conv2d(in_channels//2,in_channels,kernel_size=2,stride=2,padding=0),nn.BatchNorm2d(in_channels),nn.GELU())


        # P5_
        self.stage6_1=nn.Sequential(nn.Conv2d(in_channels*9,in_channels*3,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(in_channels*3),nn.GELU())
        self.stage6=C3(in_channels*3,in_channels,N)


        # decode
        self.de1=nn.Sequential(nn.Conv2d(in_channels,in_channels//2,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(in_channels//2),nn.GELU())
        self.de1_c3=C3(in_channels ,in_channels//2,N1)

        self.de2 = nn.Sequential(nn.Conv2d(in_channels//2, in_channels // 4, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(in_channels // 4), nn.GELU())
        self.de2_c3 = C3(in_channels // 2, in_channels // 4, N1)

        self.de3 = nn.Sequential(nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(in_channels // 8), nn.GELU())
        self.de3_c3 = C3(in_channels // 4, in_channels // 8, N1)

        # unet-like
        self.c3_1=C3(in_channels // 8, in_channels // 16, US_N)
        self.c3_2=C3(in_channels // 16, in_channels // 32, US_N)

        self.seg_head=nn.Conv2d(in_channels//32, 2, kernel_size=1, stride=1, padding=0)

        self.us=nn.Upsample(scale_factor=2,mode="bilinear")
        self.convd2x_1    = UpsampleConvLayer(in_channels, in_channels, kernel_size=4, stride=2)
        self.convd2x_2    = UpsampleConvLayer(in_channels//2, in_channels//2, kernel_size=4, stride=2)
        self.convd2x_3    = UpsampleConvLayer(in_channels//4, in_channels//4, kernel_size=4, stride=2)
        self.convd2x_4    = UpsampleConvLayer(in_channels//8, in_channels//8, kernel_size=4, stride=2)
        self.convd2x_5    = UpsampleConvLayer(in_channels//16, in_channels//16, kernel_size=4, stride=2)


    def forward(self, P2, P3, P4, P5):# [2 64 64 64][2 128 32 32][2 256 16 16][2 512 8 8]
        P2downsample1=self.P2downsample1(P2) # [2 128 32 32]
        P2downsample2=self.P2downsample2(P2) # [2 256 16 16]
        P2downsample3=self.P2downsample3(P2) # [2 512 8 8]

        P3downsample1=self.P3downsample1(P3) # [2 256 16 16]
        P3downsample2=self.P3downsample2(P3) # [2 512 8 8]
        P3upsample1=self.us(self.P3us1(P3)) # [2 64 64 64]

        P4downsample1=self.P4downsample1(P4)
        P4upsample1=self.us(self.P4us1(P4))
        P4upsample2=self.us(self.us(self.P4us2(P4)))

        P5upsample1=self.us(self.P5us1(P5))
        P5upsample2=self.us(self.us(self.P5us2(P5)))
        P5upsample3=self.us(self.us(self.us(self.P5us3(P5))))


        # P4_
        P4_=torch.cat([P5upsample1,P4,P3downsample1,P2downsample2],dim=1)
        P4_=self.stage1(self.stage1_1(P4_))
        P4_upsample1=self.us(self.P4_us1(P4_))
        P4_upsample2=self.us(self.us(self.P4_us2(P4_)))
        P4_downsample1=self.P4_downsample1(P4_)


        # P3_
        P3_=torch.cat([P4_upsample1,P5upsample2,P4upsample1,P3,P2downsample1],dim=1)
        P3_=self.stage2(self.stage2_1(P3_))
        P3_downsample1=self.P3_downsample1(P3_)
        P3_downsample2=self.P3_downsample2(P3_)
        P3_upsample1=self.us(self.P3_us1(P3_))

        # P2_
        P2_=torch.cat([P2,P3upsample1,P4upsample2,P5upsample3,P4_upsample2,P3_upsample1],dim=1)
        P2_=self.stage3(self.stage3_1(P2_))
        P2_downsample1=self.P2_downsample1(P2_)
        P2_downsample2=self.P2_downsample2(P2_)
        P2_downsample3=self.P2_downsample3(P2_)

        # P3__
        P3__=torch.cat([P2_downsample1,P3,P4upsample1,P5upsample2,P4_upsample1,P3_,P2_downsample1],dim=1)
        P3__ = self.stage4(self.stage4_1(P3__))
        P3__downsample1 = self.P3__downsample1(P3__)
        P3__downsample2 = self.P3__downsample2(P3__)


        # P4__
        P4__=torch.cat([P2downsample2,P3downsample1,P4,P5upsample1,P4_,P3_downsample1,P2_downsample2,P3__downsample1],dim=1)
        P4__=self.stage5(self.stage5_1(P4__))
        P4__downsample1=self.P4__downsample1(P4__)

        # P5_
        P5_=torch.cat([P2downsample3,P3downsample2,P4downsample1,P5,P4_downsample1,P3_downsample2,P2_downsample3,P3__downsample2,P4__downsample1],dim=1)
        P5_=self.stage6(self.stage6_1(P5_))# [2 512 8 8]
        
        ####################################################################
        #
        #     2024.11.30���ָ��ֱ��ʲ��ֵ��ϲ��������޸ĳ���ת�þ��
        #
        #################################################################
        
        out=self.convd2x_1(P5_) # [2 512 16 16]
        out=self.de1(out)# [2 256 16 16]
        out=self.de1_c3(torch.cat([out,P4__],dim=1))# [2 256 16 16]
        out=self.convd2x_2(out)# [2 256 32 32]
        out=self.de2(out)# [2 128 32 32]
        out=self.de2_c3(torch.cat([out,P3__],dim=1))# [2 128 32 32]
        out = self.convd2x_3(out)# [2 128 64 64]
        out = self.de3(out)# [2 64 64 64]
        out = self.de3_c3(torch.cat([out, P2_], dim=1))# [2 64 64 64]
        
        out=self.c3_1(self.convd2x_4(out))# [2 32 128 128]
        out=self.c3_2(self.convd2x_5(out))# [2 16 256 256]

        out=self.seg_head(out)# [2 2 256 256]

        return out



class Model_seg2(nn.Module):
    def __init__(self, dims=[64, 128, 256, 512],nums=[4,4,5,4],in_channels=512,N_mlp=4,N_decode=4,N_FPN=2,us_n=3):
        super(Model_seg2_, self).__init__()
        self.encode=Encoder(embed_dims=dims, depths=nums,N_C=N_mlp)

        self.fuse0=Fusion_Block(dims[0])
        self.fuse1=Fusion_Block(dims[1])
        self.fuse2=Fusion_Block(dims[2])
        self.fuse3=Fusion_Block(dims[3])

        self.refine1=Refinement_module(dims[2])
        self.refine2=Refinement_module(dims[1])
        self.refine3 = Refinement_module(dims[0])

        self.decode=Dense_Decode(in_channels,3,1,N_decode,N_FPN,us_n)

    def forward(self, x1, x2):
        res=[]
        outs1,outs2=self.encode(x1,x2)
        x1_0,x1_1,x1_2,x1_3=outs1
        x2_0,x2_1,x2_2,x2_3=outs2

        P2=self.fuse0(x1_0,x2_0)
        P3=self.fuse1(x1_1,x2_1)
        P4=self.fuse2(x1_2,x2_2)
        P5=self.fuse3(x1_3,x2_3)

        P4=self.refine1(P4,P5)
        P3=self.refine2(P3,P4)
        P2=self.refine3(P2,P3)

        out=self.decode(P2,P3,P4,P5)
        res.append(out)
        return res











