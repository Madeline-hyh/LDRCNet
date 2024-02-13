import sys, os
base_path = os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__)))
sys.path.append(base_path)

from model.deform_conv import DeformConv2d
from model.networks import *
from model.Transformer import *
from model.attention import *
from model.Vip import *
import warnings
import math
warnings.filterwarnings("ignore")

# ##########################################################################
# ##---------- Resizing Modules ----------
## Resizing modules
class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

    def forward(self, x):
        x = self.up(x)
        return x
#################################################################
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='BiasFree'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
##########################################################################
def conv(in_channels, out_channels):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3,
        padding=1, stride=1)

class Conv_Block(nn.Module):
    def __init__(self, inc, outc, act=nn.GELU()):
        super(Conv_Block, self).__init__()
        modules_body = []
        modules_body.append(conv(inc, outc))
        modules_body.append(act)

        self.body = nn.Sequential(*modules_body)
        self.att = ChannelAttention(num_feat=outc)

    def forward(self, x):
        res = self.body(x)
        res = res + self.att(res)
        return res

class Base_Block(nn.Module):
    def __init__(self, inc, outc , DW_Expand=1, FFN_Expand=2):
        super().__init__()
        dw_channel = inc * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=inc, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        # Channel Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )

        self.GELU = nn.GELU()

        ffn_channel = FFN_Expand * inc
        self.conv4 = nn.Conv2d(in_channels=inc, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=inc, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv6 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.beta = nn.Parameter(torch.zeros((1, inc, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, inc, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.conv1(x)
        x = self.conv2(x)
        x = x * self.se(x)
        x = self.GELU(x)
        x = self.conv3(x)
        x = self.GELU(x)
        y = inp + x * self.beta
        x = self.conv4(y)
        x = self.GELU(x)
        x = self.conv5(x)
        x = self.GELU(x)

        x = y + x * self.gamma

        x = self.conv6(x)
        x = self.GELU(x)
        return x
    
class Deform_Block(nn.Module):
    def __init__(self, inc, outc,act=nn.GELU()):
        super(Deform_Block, self).__init__()
        modules_body = []
        modules_body.append(DeformConv2d(inc, outc))
        modules_body.append(act)
        self.body = nn.Sequential(*modules_body)
        self.att = ChannelAttention(num_feat=outc)

    def forward(self, x):
        res = self.body(x)
        res = res + self.att(res)
        return res

class ChannelAttention(nn.Module):

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        if num_feat<=8:
            squeeze_factor = 8
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.GELU(),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

    
# multi scale  module
class ASPP(nn.Module):

    def __init__(self, dim,in_dim):
        super(ASPP, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim,in_dim , 3,padding=1),LayerNorm(in_dim),
             nn.GELU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=1), LayerNorm(down_dim), nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), LayerNorm(down_dim), nn.GELU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), LayerNorm(down_dim), nn.GELU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), LayerNorm(down_dim), nn.GELU())
        self.conv5 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=1),LayerNorm(down_dim), nn.GELU())
        self.catt = ChannelAttention(num_feat=5 * down_dim)
        self.fuse = nn.Sequential(nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), LayerNorm(in_dim), nn.GELU())

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = self.conv5(x)
        #conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear')
        fuse_conv = torch.cat((conv1, conv2, conv3,conv4, conv5), 1)
        fuse_conv = fuse_conv + self.catt(fuse_conv)
        return self.fuse(fuse_conv)
       
#############################################
class Degradation_Encoder(nn.Module):
    def __init__(self, in_c, n_feat):
        super(Degradation_Encoder, self).__init__()
        self.img_to_feat = Base_Block(in_c, n_feat//8)
        self.aspp1 = ASPP(dim=n_feat//8,in_dim=n_feat//8)
        encoder_level1 = []
        encoder_level1.append(Base_Block(n_feat//8, n_feat//4))
        encoder_level1.append(ASPP(dim=n_feat//4,in_dim=n_feat//4))

        self.compress1 = nn.Conv2d(in_channels=n_feat//4,out_channels=n_feat//8,kernel_size=1)
        self.deform1 = Deform_Block(n_feat//8,n_feat//8)
        self.deform1 = Deform_Block(n_feat//8,n_feat//8)

        encoder_level2 = []
        encoder_level2.append(Base_Block(n_feat//4, n_feat//2))
        encoder_level2.append(ASPP(dim=n_feat//2,in_dim=n_feat//2))

        self.compress2_1 = nn.Conv2d(in_channels=n_feat//2,out_channels=n_feat//4,kernel_size=1)
        self.deform2 = Deform_Block(n_feat//4,n_feat//4)

        encoder_level3 = []
        encoder_level3.append(Base_Block(n_feat//2, n_feat))

        self.encoder_level1 = nn.Sequential(*encoder_level1)
        self.encoder_level2 = nn.Sequential(*encoder_level2)
        self.encoder_level3 = nn.Sequential(*encoder_level3)

        self.down12 = DownSample()#n_feat//4
        self.down23 = DownSample()#n_feat//2

    def forward(self, x):
        x = self.img_to_feat(x)  # (1, 3, 256, 256)-->(1, 8, 256, 256)
        enc1 = self.encoder_level1(x)#(1, 8, 256, 256)-->(1, 16, 256, 256)
        reduce_enc1 = self.compress1(enc1)
        enc1 = torch.cat([self.deform1(reduce_enc1),reduce_enc1],1)
        x = self.down12(enc1)  # (1, 16, 256, 256)-->(1, 16, 128, 128)
        enc2 = self.encoder_level2(x)# (1, 16, 128, 128)-->(1, 32, 128, 128)
        reduce_enc2 = self.compress2_1(enc2)#16
        enc2 = torch.cat([self.deform2(reduce_enc2),reduce_enc2],1)#32
        x = self.down23(enc2)  # (1, 32, 128, 128)-->[1, 32, 64, 64])
        enc3 = self.encoder_level3(x)  # [b,n_feat,h//4,w//4]
        return enc1,enc2,enc3


class Inter_Block(nn.Module):
    def __init__(self,scale, channels):
        super(Inter_Block, self).__init__()

        self.Large2Small = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1,bias=True),
            nn.Upsample(scale_factor=1.0/scale, mode='bilinear', align_corners=False)
        )
        self.Small2Large = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1,bias=True),
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
        )
        self.SmallConvSq = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
        )
        self.LargeConvSq = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
        self.ReLU = nn.GELU()
        self.up =  nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
        self.conv_out = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
        self.catt = ChannelAttention(num_feat=channels)
        
        self.conv1 = nn.Sequential(
           nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0, stride=1, groups=1,bias=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1,
                               groups=channels,bias=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0, stride=1, groups=1,bias=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0, stride=1, groups=1,bias=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1,
                               groups=channels,bias=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0, stride=1, groups=1,bias=True)
        )
        self.catt_conv1 = ChannelAttention(num_feat=channels)
        self.catt_input = ChannelAttention(num_feat=channels)
        self.catt_out = ChannelAttention(num_feat=channels)
        self.catt_conv2 = ChannelAttention(num_feat=channels)
        self.conv_out_f = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
        self.catt_f = ChannelAttention(num_feat=channels)
        
    def forward(self, xs, xl):
        buffer_Small1 = xs
        buffer_Small2 = self.ReLU(self.Large2Small(xl))
        buffer_Large1 = xl
        buffer_Large2 = self.ReLU(self.Small2Large(xs))
        buffer_s = torch.cat((buffer_Small1, buffer_Small2), 1)###2*channels
        buffer_l = torch.cat((buffer_Large1, buffer_Large2), 1)###2*channels
        out_s = self.ReLU(self.SmallConvSq(buffer_s)) + xs
        out_l = self.ReLU(self.LargeConvSq(buffer_l)) + xl###channels
        out = torch.cat([self.up(out_s),out_l],1)###2*channels
        out = self.conv_out(out)#compress channels
        out = out + self.catt(out)###channels
        conv1 = self.conv1(out)
        out1_down_to_up = self.catt_conv1(conv1)
        out1_up = out + out1_down_to_up
        out1_down = self.catt_input(out) + conv1
        conv2 = self.conv2(out1_up)
        out2_up = self.catt_out(out1_down) + conv2
        out2_down = self.catt_conv2(conv2) + out1_down
        out_f = torch.cat([out2_up,out2_down],1)
        out_f = self.conv_out_f(out_f)
        out_f = out_f + self.catt_f(out_f)###channels
        return out_f
     
_Generation_Feature = 48;
_Generation_Out_Feature = 48;

class Rain_Generation(nn.Module):
    '''
    seg_dim:input width
    '''
    def __init__(self, n_feat=64):
        super(Rain_Generation, self).__init__()
        self.img_to_feat = Base_Block(3, _Generation_Feature)

        self.res1 = Base_Block(_Generation_Feature, _Generation_Feature)  
        self.res2 = Base_Block(_Generation_Feature, _Generation_Feature)
        self.res3 = Base_Block(_Generation_Feature, _Generation_Feature)
        # inter degradation into
        #need to convert degradation to the same channels
        self.conv4_degradation = Base_Block(n_feat//4, _Generation_Feature)
        self.conv5_degradation = Base_Block(n_feat//2, _Generation_Feature)
        self.conv6_degradation = Base_Block(n_feat, _Generation_Feature)
        self.inter4 = Inter_Block(1, _Generation_Feature)#scale:4 means width ratio
        self.inter5 = Inter_Block(2, _Generation_Feature)
        self.inter6 = Inter_Block(4, _Generation_Feature)
        self.res4 = Base_Block(_Generation_Feature, _Generation_Feature)
        self.res5 = Base_Block(_Generation_Feature, _Generation_Feature)
        self.res6 = Base_Block(_Generation_Feature, _Generation_Feature)

        self.cat_conv = Base_Block(_Generation_Feature * 6, _Generation_Out_Feature)
        self.out_conv = Base_Block(_Generation_Out_Feature, 3)

    def forward(self, x, deg1,deg2,deg3):
        '''
        deg1:(1, 16, 256, 256)
        deg2:(1, 32, 128, 128)
        deg3:(1, 64, 64, 64)
        '''
        x_feat = self.img_to_feat(x)

        x_res1 = x_feat + self.res1(x_feat)
        x_res2 = x_res1 + self.res2(x_res1)
        x_res3 = x_res2 + self.res3(x_res2)
        x_res3 = self.inter4(self.conv4_degradation(deg1), x_res3)
        x_res4 = x_res3 + self.res4(x_res3)
        x_res4 = self.inter5(self.conv5_degradation(deg2),x_res4)
        x_res5 = x_res4 + self.res5(x_res4)
        x_res5 = self.inter6(self.conv6_degradation(deg3),x_res5)
        x_res6 = x_res5 + self.res6(x_res5)

        x_cat = torch.cat([x_res1, x_res2, x_res3, x_res4, x_res5, x_res6], dim=1)
        return x + self.out_conv(self.cat_conv(x_cat))

class SEModel(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Derain(nn.Module):
    def __init__(self, n_feat,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 ):
        super(Derain, self).__init__()
        self.image_to_feat = Base_Block(inp_channels, dim//2)
        self.deform1 = Deform_Block(inc=dim//2,outc=dim//2)
        self.image_to_feat2 = Base_Block(dim//2, dim)
        self.aspp1 = ASPP(dim=dim,in_dim=dim)

        encoder_level1 = []
        encoder_level1.append(Base_Block(inc=dim, outc=dim*2))
        encoder_level1.append(ASPP(dim=dim*2,in_dim=dim*2))
        self.encoder_level1 = nn.Sequential(*encoder_level1)

        self.down1_2 = DownSample()  ## From Level 1 to Level 2 #dim*2
        self.reduce2_1 = nn.Conv2d(in_channels=dim*2,out_channels=dim//2,kernel_size=1)
        self.deform2 = Deform_Block(inc=dim//2,outc=dim//2)
        self.reduce2_2 = nn.Conv2d(in_channels=dim*2,out_channels=dim,kernel_size=1)

        encoder_level2 = []
        encoder_level2.append(Base_Block(inc=int(dim * 2), outc=int(dim * 2 ** 2)))
        encoder_level2.append(ASPP(dim=dim*2**2,in_dim=dim*2**2))
        self.encoder_level2 = nn.Sequential(*encoder_level2)

        self.down2_3 = DownSample()  ## From Level 2 to Level 3 #dim*2**2

        # self.latent =  nn.Sequential(*[
        # SEModel(int(dim * 2 ** 2)),
        # Base_Block(inc=int(dim * 2 ** 2), outc=int(dim * 2)),
        # ASPP(dim=int(dim * 2),in_dim=int(dim * 2)),
        # Base_Block(inc=int(dim * 2),outc=int(dim * 2 ** 2))
        # ])
        self.latent = nn.Sequential(*[TransformerBlock(int(dim * 2 ** 2)) for i in range(2)])

        self.up3_2 = UpSample()  ## From Level 3 to Level 2 #dim*2**2
        decoder_level2 = []
        decoder_level2.append(Base_Block(inc=int(dim * 2 ** 2), outc=int(dim * 2)))
        self.decoder_level2 = nn.Sequential(*decoder_level2)

        self.up2_1 = UpSample()  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels) #dim*2
        decoder_level1 = []
        decoder_level1.append(Base_Block(inc=int(dim * 2 ** 1), outc=dim))
        self.decoder_level1 = nn.Sequential(*decoder_level1)
     
        # inter degradation into
        self.conv3_degradation = Base_Block(n_feat//2, int(dim * 2 ** 2))
        self.conv2_degradation = Base_Block(n_feat//2, int(dim * 2))
        self.conv1_degradation = Base_Block(n_feat//4, int(dim * 2))
        self.conv0_degradation = Base_Block(n_feat//4, dim)

        self.inter3 = Inter_Block(1, int(dim * 2 ** 2))
        self.inter2 = Inter_Block(1, int(dim * 2))
        self.inter1 = Inter_Block(1, int(dim * 2))
        self.inter0 = Inter_Block(1, dim)

        self.output = Base_Block(dim, out_channels)

    def forward(self, inp_img,deg1,deg2,deg3):
        '''
        deg1:(1, 16, 256, 256)
        deg2:(1, 32, 128, 128)
        deg3:(1, 64, 64, 64)
        '''
        inp_enc_level1 = self.image_to_feat(inp_img)# (1, 3, 256, 256)-->(1, dim//2, 256, 256)
        deform1 = self.deform1(inp_enc_level1)# dim//2
        inp_enc_level1 = self.image_to_feat2(deform1)#dim
        inp_enc_level1 = self.aspp1(inp_enc_level1)#dim

        out_enc_level1 = self.encoder_level1(inp_enc_level1)# (1, dim, 256, 256)-->(1, dim*2, 256, 256)

        inp_enc_level2 = self.down1_2(out_enc_level1)# (1, dim*2, 256, 256)-->(1, dim*2, 128, 128)
        reduce_enc_level2 = self.reduce2_1(inp_enc_level2)#dim//2
        deform2 = self.deform2(reduce_enc_level2)#dim//2
        temp_enc_level2 = torch.cat([reduce_enc_level2,deform2],1)#dim
        ori_enc_level2 = self.reduce2_2(inp_enc_level2)#dim
        inp_enc_level2 = torch.cat([temp_enc_level2,ori_enc_level2],1)#dim*2
        out_enc_level2 = self.encoder_level2(inp_enc_level2)# (1, dim*2, 128, 128)-->(1, dim*2**2, 128, 128)

        inp_enc_level3 = self.down2_3(out_enc_level2)# (1, dim*2**2, 128, 128)-->(1, dim*2**2, 64, 64)
        latent = self.latent(inp_enc_level3)  # dim=int(dim*2**2)

        inp_dec_level2 = self.up3_2(latent)# (1, dim*2**2, 64, 64)-->(1, dim*2**2, 128, 128)

        inp_dec_level2 = self.inter3(self.conv3_degradation(deg2), inp_dec_level2)
        inp_dec_level2 = inp_dec_level2 + out_enc_level2#(1, dim*2**2, 128, 128)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)# (1, dim*2**2, 128, 128)-->(1, dim*2, 128, 128)

        out_dec_level2 = self.inter2(self.conv2_degradation(deg2), out_dec_level2)#(1, dim*2, 128, 128)
       

        inp_dec_level1 = self.up2_1(out_dec_level2)# (1, dim*2, 128, 128)-->(1, dim*2, 256, 256)
        inp_dec_level1 = self.inter1(self.conv1_degradation(deg1), inp_dec_level1)#(1, dim*2, 256, 256)
        inp_dec_level1 = inp_dec_level1 + out_enc_level1
        out_dec_level1 = self.decoder_level1(inp_dec_level1)  #(1, dim*2, 256, 256)-->(1, dim, 256, 256)
        
        out_dec_level1 = self.inter0(self.conv0_degradation(deg1), out_dec_level1)#(1, dim, 256, 256)

        output = self.output(out_dec_level1)
        return (inp_img - output).clamp(min=0,max=1),deg1,deg2,deg3

##########################################################################
class LDRCNet(nn.Module):
    def __init__(self, in_c=3, n_feat=64):
        super(LDRCNet, self).__init__()
        self.encoder = Degradation_Encoder(in_c, n_feat)
        self.derain = Derain(n_feat)

    def forward(self, input):
        deg1,deg2,deg3 = self.encoder(input)
        output,deg1,deg2,deg3= self.derain(input,deg1,deg2,deg3)
        return output, deg1, deg2, deg3


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == '__main__':
    input = torch.rand(size=(1, 3, 256, 256)).cpu()
    model_derain =  LDRCNet().cpu()
    # derain_output,deg1,deg2,deg3 = model_derain(input)
    # print(derain_output.size())
    # from torchstat import stat
    # print("Derain_Para:",stat(model_derain,(3,256,256)))
    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    # flops = FlopCountAnalysis(model_derain, input)
    # print("FLOPs:", flops.total())
    # print("parameters:", count_parameters(model_derain))


    input_size =256
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model_derain, (3, input_size, input_size), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # import time
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # model_derain = model_derain.to(device)  # 网络模型
    # input = input.to(device)  # 输入
    # torch.cuda.synchronize()
    # time_start = time.time()
    # derain_output,deg1,deg2,deg3 = model_derain(input)
    # torch.cuda.synchronize()
    # time_end = time.time()
    # time_sum = time_end - time_start
    # print(time_sum)