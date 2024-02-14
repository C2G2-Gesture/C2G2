import math
import torch
import torch.nn as nn
import numpy as np
from model.diffusion_util import TransformerModel, TransformerEncoder, PositionalEncoding

def nonlinearity(x):
    # swish
    act = nn.LeakyReLU(0.2,True)
    return act(x)


def Normalize(in_channels):
    # return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    return torch.nn.BatchNorm1d(in_channels)

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512,downsample=True):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.downsample = downsample

        self.norm1 = Normalize(in_channels)
        if self.downsample:
            self.conv1 = torch.nn.Conv1d(in_channels,out_channels,3,2,1)
        else:
            self.conv1 = torch.nn.Conv1d(in_channels,out_channels,3,1,1)
            
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,3,1,1)
        
        if self.downsample:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,3,2,1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,3,2,1)
        else:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,3,1,1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,3,1,1)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        
        if self.use_conv_shortcut:
            x = self.conv_shortcut(x)
        else:
            x = self.nin_shortcut(x)

        return x+h
    
class TransResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512,downsample=True):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.downsample = downsample

        self.norm1 = Normalize(in_channels)
        if self.downsample:
            self.conv1 = torch.nn.ConvTranspose1d(in_channels,out_channels,3,2,1,1)
        else:
            self.conv1 = torch.nn.ConvTranspose1d(in_channels,out_channels,3,1,1,0)
            
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.ConvTranspose1d(out_channels,
                                     out_channels,3,1,1,0)
        
        if self.downsample:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.ConvTranspose1d(in_channels,
                                                    out_channels,3,2,1,1)
            else:
                self.nin_shortcut = torch.nn.ConvTranspose1d(in_channels,
                                                    out_channels,3,2,1,1)
        else:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.ConvTranspose1d(in_channels,
                                                    out_channels,3,1,1,0)
            else:
                self.nin_shortcut = torch.nn.ConvTranspose1d(in_channels,
                                                    out_channels,3,1,1,0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # if self.in_channels != self.out_channels:
        if self.use_conv_shortcut:
            x = self.conv_shortcut(x)
        else:
            x = self.nin_shortcut(x)

        return x+h
    
class Encoder_convd(nn.Module):
    def __init__(self, *, ch, ch_mult=(1,2,2,4), 
                 downsample_rate=4,dropout=0.0, resamp_with_conv=True, in_channels,
                 z_channels, depth = 4,double_z=False, pos_enc = False, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.in_channels = in_channels
        self.depth = depth
        self.downsample_rate = downsample_rate-1
        
        # assert len(ch_mult)==downsample_rate
            
        # downsampling
        self.conv_in = torch.nn.Conv1d(in_channels,ch,3,1)

        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        
        for i_level in range(self.downsample_rate):
            block = nn.ModuleList()

            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            
            block.append(ResnetBlock(in_channels=block_in,
                                        out_channels=block_out,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout,
                                        downsample=True))
            
            block.append(ResnetBlock(in_channels=block_out,
                                        out_channels=block_out,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout,
                                        downsample=False))
            
            block_in = block_out
            
            down = nn.Module()
            down.block = block
            # if i_level != self.num_resolutions-1:
            #     down.downsample = Downsample(block_in, resamp_with_conv)
            #     curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       downsample=False)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        2*z_channels if double_z else z_channels,3,1,1)
        self.out_transencoder = TransformerEncoder(num_pose = int(32//2**(downsample_rate-1)),pose_dim=z_channels,depth=self.depth)


    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)
        b,t,n = x.shape
        x = x.transpose(1,2)
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.downsample_rate):
            h = self.down[i_level].block[0](hs[-1], temb)
            hs.append(h)
            h = self.down[i_level].block[1](hs[-1], temb)
            hs.append(h)
            # h = self.down[i_level].block[1](hs[-1], temb)
            # hs.append(h)

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        
         # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = h.transpose(1,2)
        h = self.out_transencoder(h)
        
        return h


class Decoder_convd(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,2,4), downsample_rate = 4,
                 dropout=0.0, resamp_with_conv=True, 
                 z_channels, depth = 2, give_pre_end=False, pos_enc=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        
        self.in_channels = z_channels
        self.give_pre_end = give_pre_end
        self.depth = depth
        self.downsample_rate = downsample_rate-1
        # assert len(ch_mult)==downsample_rate
        
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[0]

        # z to block_in
        self.conv_in = torch.nn.Conv1d(z_channels,
                                       block_in,3,1,1) # 34 -> 32

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       downsample=False)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in (range(self.downsample_rate)):
            block = nn.ModuleList()

            block_out = ch*ch_mult[i_level]
           
            block.append(TransResnetBlock(in_channels=block_in,
                                        out_channels=block_out,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout,
                                        downsample=True))
            
            block.append(TransResnetBlock(in_channels=block_out,
                                        out_channels=block_out,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout,
                                        downsample=False))
            
            block_in = block_out
            
            up = nn.Module()
            up.block = block
            
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.ConvTranspose1d(block_in,
                                        out_ch,3,1,0)
        
        self.out_transencoder = TransformerEncoder(num_pose = int(32//2**(downsample_rate-1)),pose_dim=z_channels,depth = self.depth)

    def forward(self, z):
        b,t,n = z.shape
        temb = None
        
        h = self.out_transencoder(z)
        h = h.transpose(1,2)

        # z to block_in
        h = self.conv_in(h)

        # middle
        h = self.mid.block_1(h, temb)

        # upsampling
        for i_level in reversed(range(self.downsample_rate)):
            # for i_block in range(self.num_res_blocks+1):
            h = self.up[i_level].block[0](h, temb)
            h = self.up[i_level].block[1](h, temb)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = h.transpose(1,2)
        return h