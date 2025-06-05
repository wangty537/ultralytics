# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""ecb modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


__all__ = (
    "ECB",
    "SeqConv3x3",
)

def measure_inference_speed(model, data, max_iter=50, log_interval=10):
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i in range(max_iter):

        #torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(*data)

        #torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    return fps

class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias

        elif self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        if self.type == 'conv1x1-conv3x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        return y1

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.type == 'conv1x1-conv3x3':
            # re-param conv kernel
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            # 将输入 RB 转换为与 self.k1 相同的半精度类型
            RB = RB.to(self.k1.dtype)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1, ) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias

            # 将 k1 转换为与 self.k0 相同的半精度类型
            k1 = k1.to(self.k0.dtype)
            
            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            # 将输入 RB 转换为与 self.k1 相同的半精度类型
            RB = RB.to(k1.dtype)
            RB = F.conv2d(input=RB, weight=k1).view(-1, ) + b1
        return RK, RB


class ECB(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier, act_type='prelu', with_idt=False):
        super(ECB, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.in_channels = inp_planes
        self.out_channels = out_planes
        self.inp_planes = self.in_channels
        self.out_planes = self.out_channels
        self.act_type = act_type

        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.inp_planes, self.out_planes, -1)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.inp_planes, self.out_planes, -1)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.inp_planes, self.out_planes, -1)

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'lrelu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        #print('training: ', self.training)
        if self.training:
            y = self.conv3x3(x) + \
                self.conv1x1_3x3(x) + \
                self.conv1x1_sbx(x) + \
                self.conv1x1_sby(x) + \
                self.conv1x1_lpl(x)
            if self.with_idt:
                y += x
        else:
            RK, RB = self.rep_params()
            y = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
        if self.act_type == 'lrelu':
            y = self.lrelu(y)
        elif self.act_type != 'linear':
            y = self.act(y)
        return y

    def rep_params(self):
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        RK, RB = (K0 + K1 + K2 + K3 + K4), (B0 + B1 + B2 + B3 + B4)

        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB
    
    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt

class Conv3X3(nn.Module):
    def __init__(self, inp_planes, out_planes, act_type='prelu'):
        super(Conv3X3, self).__init__()

        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type

        self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.act = None

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'lrelu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        y = self.conv3x3(x)
        if self.act_type == 'lrelu':
            y = self.lrelu(y)
        elif self.act_type != 'linear':
            y = self.act(y)
      
        return y
    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt


class PlainSR(nn.Module):
    def __init__(self, module_nums=10, channel_nums=16, act_type='prelu', scale=2, colors=3):
        super(PlainSR, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        backbone = []
        backbone += [Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type)]
        for i in range(self.module_nums):
            backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type)]
        # backbone += [
        #     Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums , act_type='linear')]
        self.backbone = nn.Sequential(*backbone)

        self.upsampler = nn.Conv2d(self.channel_nums, self.colors, kernel_size=3, padding=1, stride=1) #self.upsampler = nn.PixelShuffle(self.scale)

    def forward(self, x):
        #y = self.backbone(x) + x
        y = self.upsampler(self.backbone(x)) +x
        return y


class ECBSR(nn.Module):
    def __init__(self, module_nums, channel_nums, with_idt=True, act_type='prelu', scale=2, colors=3):
        super(ECBSR, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.with_idt = with_idt
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        backbone = []
        backbone += [
            ECB(self.colors, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt=self.with_idt)]
        for i in range(self.module_nums):
            backbone += [ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type,
                             with_idt=self.with_idt)]
        # backbone += [
        #     ECB(self.channel_nums, self.channel_nums , depth_multiplier=2.0, act_type='linear',
        #         with_idt=self.with_idt)]

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.Conv2d(self.channel_nums, self.colors, kernel_size=3, padding=1, stride=1) #nn.PixelShuffle(self.scale)

    def forward(self, x):
        y = self.upsampler(self.backbone(x)) +x
        return y
    
class ECBSR_ps(nn.Module):
    def __init__(self, module_nums, channel_nums, with_idt=True, act_type='prelu', scale=2, colors=3):
        super(ECBSR_ps, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.with_idt = with_idt
        self.act_type = act_type


        #self.unpixelshuffle = nn.PixelUnshuffle(scale)
       
        self.pixelshuffle = nn.PixelShuffle(scale)
        self.in_channels = self.colors * scale * scale
        self.out_channels = self.colors * scale * scale
        self.unpixelshuffle = nn.Conv2d(colors, self.in_channels, kernel_size=3, padding=1, stride=2)
        self.m = ECBSR(module_nums, channel_nums, with_idt, act_type, scale, colors=self.in_channels)

    def forward(self, x):
        x = self.unpixelshuffle(x)
        x = self.m(x)
        y = self.pixelshuffle(x)
        
        return y
class PlainSR_ps(nn.Module):
    def __init__(self, module_nums=10, channel_nums=16, act_type='prelu', scale=2, colors=3):
        super(PlainSR_ps, self).__init__()

        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        #self.with_idt = with_idt
        self.act_type = act_type
        self.in_channels = self.colors * scale * scale
        self.out_channels = self.colors * scale * scale

        self.unpixelshuffle = nn.Conv2d(colors, self.in_channels, kernel_size=3, padding=1, stride=2) #nn.PixelUnshuffle(scale)
        self.pixelshuffle = nn.PixelShuffle(scale)

        self.m = PlainSR(module_nums, channel_nums, act_type, scale, colors=self.in_channels)
    def forward(self, x):
        #y = self.backbone(x) + x
        x = self.unpixelshuffle(x)
        x = self.m(x)
        y = self.pixelshuffle(x)
        return y    
class ECBSR_ps_conv(nn.Module):
    def __init__(self, module_nums=10, channel_nums=16, with_idt=True, act_type='prelu', scale=2, colors=3):
        super(ECBSR_ps_conv, self).__init__()

        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.with_idt = with_idt
        self.act_type = act_type


        #self.unpixelshuffle = nn.PixelUnshuffle(scale)
       
        self.pixelshuffle = nn.PixelShuffle(scale)
        self.in_channels = self.colors * scale * scale
        self.out_channels = self.colors * scale * scale
        self.unpixelshuffle = nn.Conv2d(colors, self.in_channels, kernel_size=3, padding=1, stride=2)
        self.m = ECBSR(module_nums, channel_nums, with_idt, act_type, scale, colors=self.in_channels)
        self.conv = ECB(self.colors, self.colors, depth_multiplier=2.0, act_type='linear', with_idt=self.with_idt)
    def forward(self, x):
        #x = self.unpixelshuffle(x)
        #x = torch.reshape(x, (-1, self.colors, 200, 2,200,2)).permute(0, 1, 3, 5, 2,4).reshape(-1, 12, 200,200)
        #x = F.conv2d(x, self.w, bias=None, padding=0, stride=2)
        x = self.unpixelshuffle(x)
        x = self.m(x)
        y = self.pixelshuffle(x)

        y = self.conv(y)
        
        return y
    
class ECBSR_ps_trans_conv(nn.Module):
    def __init__(self, module_nums=10, channel_nums=16, with_idt=True, act_type='prelu', scale=2, colors=3):
        super(ECBSR_ps_trans_conv, self).__init__()

        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.with_idt = with_idt
        self.act_type = act_type


        #self.unpixelshuffle = nn.PixelUnshuffle(scale)
       
       
        self.unpixelshuffle = nn.Conv2d(colors, self.channel_nums, kernel_size=3, padding=1, stride=2)
        
        backbone = []
        backbone += [
            ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt=self.with_idt)]
        for i in range(self.module_nums):
            backbone += [ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type,
                             with_idt=self.with_idt)]
        # backbone += [
        #     ECB(self.channel_nums, self.channel_nums , depth_multiplier=2.0, act_type='linear',
        #         with_idt=self.with_idt)]

        self.m = nn.Sequential(*backbone)
        
        self.up = nn.ConvTranspose2d(channel_nums*2, channel_nums, 2, stride=2)
        #self.conv = nn.Conv2d(channel_nums, 3, kernel_size=3, padding=1, stride=1 )
        
        self.conv = ECB(self.channel_nums, self.colors, depth_multiplier=2.0, act_type='linear', with_idt=self.with_idt)
    def forward(self, x):
        #x = self.unpixelshuffle(x)
        #x = torch.reshape(x, (-1, self.colors, 200, 2,200,2)).permute(0, 1, 3, 5, 2,4).reshape(-1, 12, 200,200)
        #x = F.conv2d(x, self.w, bias=None, padding=0, stride=2)
        x1 = self.unpixelshuffle(x)
        x2 = self.m(x1)
        x3 = torch.cat([x1, x2], 1)
        x3 = self.up(x3)
       

        y = self.conv(x3)
        
        return y
class ECBSR_ps_interp_conv(nn.Module):
    def __init__(self, module_nums=10, channel_nums=16, with_idt=True, act_type='prelu', scale=2, colors=3):
        super(ECBSR_ps_interp_conv, self).__init__()

        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.with_idt = with_idt
        self.act_type = act_type


        #self.unpixelshuffle = nn.PixelUnshuffle(scale)
       
       
        self.unpixelshuffle = nn.Conv2d(colors, self.channel_nums, kernel_size=3, padding=1, stride=2)
        
        backbone = []
        backbone += [
            ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt=self.with_idt)]
        for i in range(self.module_nums):
            backbone += [ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type,
                             with_idt=self.with_idt)]
        # backbone += [
        #     ECB(self.channel_nums, self.channel_nums , depth_multiplier=2.0, act_type='linear',
        #         with_idt=self.with_idt)]

        self.m = nn.Sequential(*backbone)
        
        self.up = ECB(self.channel_nums*2, self.channel_nums, depth_multiplier=2.0, act_type='linear', with_idt=self.with_idt) #nn.Conv2d(channel_nums*2, channel_nums, kernel_size=3, padding=1, stride=1)
        #self.conv = nn.Conv2d(channel_nums, 3, kernel_size=3, padding=1, stride=1 )
        
        self.conv = ECB(self.channel_nums, self.colors, depth_multiplier=2.0, act_type='linear', with_idt=self.with_idt)
    def forward(self, x):
        #x = self.unpixelshuffle(x)
        #x = torch.reshape(x, (-1, self.colors, 200, 2,200,2)).permute(0, 1, 3, 5, 2,4).reshape(-1, 12, 200,200)
        #x = F.conv2d(x, self.w, bias=None, padding=0, stride=2)
        x1 = self.unpixelshuffle(x)
        x2 = self.m(x1)
        x3 = torch.cat([x1, x2], 1)
        x3 = self.up(x3)
       
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear')
        y = self.conv(x3)
        
        return y

import torch
import torch.nn as nn
import torch.nn.functional as F



class PlainSR_dnet(nn.Module):
    def __init__(self, module_nums=10, channel_nums=16, act_type='prelu', scale=2, colors=3):
        super(PlainSR_dnet, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        backbone = []
        backbone += [Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type)]
        for i in range(self.module_nums):
            backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type)]
        # backbone += [
        #     Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums , act_type='linear')]
        self.backbone = nn.Sequential(*backbone)

        self.upsampler = nn.Conv2d(self.channel_nums, 1, kernel_size=3, padding=1, stride=1) #self.upsampler = nn.PixelShuffle(self.scale)

    def forward(self, x):
        #y = self.backbone(x) + x
        y = self.upsampler(self.backbone(x)) +x
        return y
        

    
    
def ecbsr2plain(PATH = '/home/redpine/code/experiments_400gamma/unetsrgbcpu_lize400_n2c_ft_ps256_bs32_enhance0_ecbsr_m4c8_gan0_cutblur1_distill0_20240318_122800/models/ecbsrm4c8_gan0_cutblur1_distill0_checkpoint_10000.pth',
                siz = 400, ):
        # ecbsr to plain
    # Create the model
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)

    model1 = ECBSR(4, 8, act_type='prelu')
    model2 = PlainSR(4, 8, act_type='prelu')
    model1.eval()

    checkpoint = torch.load(PATH, map_location=device)
    model1.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})

    state_dict = model1.state_dict()
    # state_dict['conv2d_sharp.weight'] = nn.Parameter(data=model1.conv2_1.rep_params(), requires_grad=False)
    model2.load_state_dict(state_dict, strict=False)
    model2.eval()
    
    for name, param in model1.named_parameters():
        print(name, param.shape)
    
    print('-'*50)
    for name, param in model2.named_parameters():
        print(name, param.shape)
    for i in range(len(model2.backbone)):
        model2.backbone[i].conv3x3.weight, model2.backbone[i].conv3x3.bias = nn.Parameter(model1.backbone[i].rep_params()[0], requires_grad=False), nn.Parameter(model1.backbone[i].rep_params()[1], requires_grad=False)
    
    torch.save(model2.state_dict(), PATH[:-4] + "_" + str(siz) + "_plain.pth")
    model = model2.to(device)
    model.eval()

    # Create input
    x = torch.randn((1, 3, siz, siz)).to(device)  # .cuda()    # input format: nchw

    # Export to onnx model
    print("Converting to onnx...")
    torch.onnx.export(model,  # model being run
                      x,  # model input
                      PATH[:-4] + "_" + str(siz) + "_plain.onnx",  # where to save the model
                      opset_version=18,  # the ONNX version to export the model to
                      do_constant_folding=True,  # use constant folding optimization
                      input_names=['input'],  # specify the model input names
                      output_names=['output'],  # specify the model output names
                      )
    print("Done! Onnx output model: exbsr.onnx")
    model1 = model1.to(device)
    model2 = model2.to(device)
    x = torch.rand(1, 3, 400, 400).to(device)*100
    y = model1(x)
    y1 = model2(x)
    print(y.shape, y1.shape)
    t = y1 - y
    print(t[0, 1, 100:110,100:110])
    print(torch.sum(t))

    fps = measure_inference_speed(model1, [x])
    print('fps:', fps, 1 / fps)
    fps = measure_inference_speed(model2, [x])
    print('fps:', fps, 1 / fps)
    return model

def ecbsr2plain_ps(PATH = '/home/redpine/code/experiments_400_v1000_siddpretrain_mix/unetsrgbcpu_6946_mix_n2c_ft_ps256256_bs1632_enhance0_ECBSR_ps256_m6c32_gan0_cutblur1_distill0_20240418_203310/models/ecbsrps_m6c32_checkpoint_5000.pth',
                siz = 400, ):
        # ecbsr to plain
    # Create the model
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)

    model1 = ECBSR_ps(6, 32, act_type='prelu')
    model2 = PlainSR_ps(6, 32, act_type='prelu')
    model1.eval()

    checkpoint = torch.load(PATH, map_location=device)
    model1.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})

    state_dict = model1.state_dict()
    # state_dict['conv2d_sharp.weight'] = nn.Parameter(data=model1.conv2_1.rep_params(), requires_grad=False)
    model2.load_state_dict(state_dict, strict=False)
    model2.eval()
    
    for name, param in model1.named_parameters():
        print(name, param.shape)
    
    print('-'*50)
    for name, param in model2.named_parameters():
        print(name, param.shape)
    for i in range(len(model2.m.backbone)):
        model2.m.backbone[i].conv3x3.weight, model2.m.backbone[i].conv3x3.bias = nn.Parameter(model1.m.backbone[i].rep_params()[0], requires_grad=False), nn.Parameter(model1.m.backbone[i].rep_params()[1], requires_grad=False)
    
    torch.save(model2.state_dict(), PATH[:-4] + "_" + str(siz) + "_plain.pth")
    model = model2.to(device)
    model.eval()

    # Create input
    x = torch.randn((1, 3, siz, siz)).to(device)  # .cuda()    # input format: nchw

    # Export to onnx model
    print("Converting to onnx...")
    torch.onnx.export(model,  # model being run
                      x,  # model input
                      PATH[:-4] + "_" + str(siz) + "_plain.onnx",  # where to save the model
                      opset_version=18,  # the ONNX version to export the model to
                      do_constant_folding=True,  # use constant folding optimization
                      input_names=['input'],  # specify the model input names
                      output_names=['output'],  # specify the model output names
                      )
    print("Done! Onnx output model: exbsr.onnx")
    model1 = model1.to(device)
    model2 = model2.to(device)
    x = torch.rand(1, 3, 400, 400).to(device)
    y = model1(x)
    y1 = model2(x)
    print(y.shape, y1.shape)
    t = y1 - y
    print(t[0, 1, 100:110,100:110])
    print(torch.mean(t.abs()))

    fps = measure_inference_speed(model1, [x])
    print('fps:', fps, 1 / fps)
    fps = measure_inference_speed(model2, [x])
    print('fps:', fps, 1 / fps)
    return model
if __name__ == '__main__':
    # # # test seq-conv
    # x = torch.randn(1, 3, 5, 5).cuda()
    # conv = SeqConv3x3('conv1x1-conv3x3', 3, 3, 2).cuda()
    # y0 = conv(x)
    # RK, RB = conv.rep_params()
    # y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    # print(y0 - y1)

    # # test ecb
    # x = torch.randn(1, 3, 5, 5).cuda() * 200
    # ecb = ECB(3, 3, 2, act_type='linear', with_idt=True).cuda()
    # y0 = ecb(x)

    # RK, RB = ecb.rep_params()
    # y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    # print(y0 - y1)
    
    # # test convert
    #ecbsr2plain()
    # net = ECBSR_ps(6, 32)
    # net = ECBSR_ps(8, 32)
    # net = ECBSR_ps_conv(6, 16)
    # net = ECBSR_ps_trans_conv(6, 16)
    # net = ECBSR_ps_interp_conv(6, 16)
    
    ecbsr2plain_ps()
    