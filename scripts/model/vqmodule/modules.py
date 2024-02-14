# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
from model.diffusion_util import TransformerModel, TransformerEncoder, PositionalEncoding

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    # return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    return torch.nn.LayerNorm(in_channels)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512,use_activation=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Linear(in_channels,
                                     out_channels)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Linear(out_channels,
                                     out_channels)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Linear(in_channels,
                                                     out_channels)
            else:
                self.nin_shortcut = torch.nn.Linear(in_channels,
                                                    out_channels)
                
        self.use_act = use_activation

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        
        if self.use_act:
            h = nonlinearity(h)
            
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        
        if self.use_act:
            h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Linear(in_channels, in_channels)
        self.k = torch.nn.Linear(in_channels, in_channels)
        self.v = torch.nn.Linear(in_channels, in_channels)

        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)


        # compute attention
        b,t,n = q.shape
        q = q.permute(0,2,1)   # b,n,t
        w_ = torch.bmm(q,k)     # b,n,n    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_




class Encoder(nn.Module):
    def __init__(self, *, ch, ch_mult=(1,2,4), num_res_blocks,
                 attn_resolutions = [], dropout=0.0, resamp_with_conv=True, in_channels,
                 z_channels, depth = 4,double_z=False, pos_enc = False,use_mid = False, use_act=False, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.depth = depth
        self.pos_enc = pos_enc
        if pos_enc:
            self.positional_encoding = PositionalEncoding(z_channels,0.1,34)
            
        # downsampling
        self.conv_in = torch.nn.Linear(in_channels,
                                       self.ch)

        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
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
                                       dropout=dropout)
        # self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Linear(block_in,
                                        2*z_channels if double_z else z_channels)
        self.out_transencoder = TransformerEncoder(pose_dim=z_channels,depth=self.depth)
        self.use_mid = use_mid
        self.use_act = use_act


    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)
        b,t,n = x.shape

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # if len(self.down[i_level].attn) > 0:
                #     h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            # if i_level != self.num_resolutions-1:
            #     hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        if self.use_mid:
            h = self.mid.block_1(h, temb)
            h = self.mid.block_2(h, temb)


        h = h.reshape(b,t,-1)
        
         # end
        h = self.norm_out(h)
        if self.use_act:
            h = nonlinearity(h)
        h = self.conv_out(h)
        # transformer middle: self_attent+ffn
        if self.pos_enc:
            x = self.positional_encoding(h)

        h = self.out_transencoder(h)
        
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,3), num_res_blocks,
                 attn_resolutions = [], dropout=0.0, resamp_with_conv=True, 
                 z_channels, depth = 2, give_pre_end=False, pos_enc=False,use_mid=False,use_act=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = z_channels
        self.give_pre_end = give_pre_end
        self.depth = depth
        self.pos_enc = pos_enc
        if pos_enc:
            self.positional_encoding = PositionalEncoding(z_channels,0.1,34)

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        # curr_res = resolution // 2**(self.num_resolutions-1)
        # self.z_shape = (1,z_channels,curr_res,curr_res)
        # print("Working with z of shape {} = {} dimensions.".format(
        #     self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Linear(z_channels,
                                       block_in)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        # self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            #     if curr_res in attn_resolutions:
            #         attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            # up.attn = attn
            # if i_level != 0:
            #     up.upsample = Upsample(block_in, resamp_with_conv)
            #     curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Linear(block_in,
                                        out_ch)
        
        self.out_transencoder = TransformerEncoder(pose_dim=z_channels,depth = self.depth)
        self.use_mid = use_mid
        self.use_act = use_act

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape
        b,t,n = z.shape
        # z = z.reshape(b*t,n)

        # timestep embedding
        temb = None
        if self.pos_enc:
            x = self.positional_encoding(z)
        
        h = self.out_transencoder(z)

        # z to block_in
        h = self.conv_in(h)

        # middle
        if self.use_mid:
            h = self.mid.block_1(h, temb)
            h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
            #     if len(self.up[i_level].attn) > 0:
            #         h = self.up[i_level].attn[i_block](h)
            # if i_level != 0:
            #     h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        if self.use_act:
            h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class ConditionalDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,3), cond_dim=64, num_res_blocks,
                 attn_resolutions = [], dropout=0.0, resamp_with_conv=True, 
                 z_channels, depth = 2, give_pre_end=False, pos_enc=False,use_mid=False,use_act=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = z_channels
        self.give_pre_end = give_pre_end
        self.depth = depth
        self.pos_enc = pos_enc
        self.use_mid = use_mid
        self.use_act = use_act
        if pos_enc:
            self.positional_encoding = PositionalEncoding(z_channels,0.1,34)

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        # curr_res = resolution // 2**(self.num_resolutions-1)
        # self.z_shape = (1,z_channels,curr_res,curr_res)
        # print("Working with z of shape {} = {} dimensions.".format(
        #     self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Linear(z_channels+cond_dim,
                                       block_in)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        # self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            #     if curr_res in attn_resolutions:
            #         attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            # up.attn = attn
            # if i_level != 0:
            #     up.upsample = Upsample(block_in, resamp_with_conv)
            #     curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Linear(block_in,
                                        out_ch)
            
        self.emb = ResnetBlock(in_channels=out_ch,
                                out_channels=cond_dim*2,
                                temb_channels=self.temb_ch,
                                dropout=dropout)
        
        self.linear_cond = nn.ModuleList()
        self.linear_cond.append(ResnetBlock(in_channels=cond_dim*2,
                                out_channels=cond_dim,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
        self.linear_cond.append(ResnetBlock(in_channels=cond_dim,
                                out_channels=cond_dim,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
        
        self.out_transencoder = TransformerEncoder(pose_dim=z_channels+cond_dim,depth = self.depth)

    def forward(self, z, cond):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape
        b,t,n = z.shape
        # z = z.reshape(b*t,n)
        
        # timestep embedding
        temb = None
        
        cond_emb = self.emb(cond,temb)
        for iblock in range(len(self.linear_cond)):
            cond_emb = self.linear_cond[iblock](cond_emb,temb)
        cond_emb = cond_emb.unsqueeze(1)
        new_emb = cond_emb
        for i in range(t-1):
            new_emb = torch.concat((new_emb,cond_emb),1)
        

        if self.pos_enc:
            x = self.positional_encoding(z)
            
        z = torch.concat((z,new_emb),2)
        
        h = self.out_transencoder(z)

        # z to block_in
        h = self.conv_in(h)

        # middle
        if self.use_mid:
            h = self.mid.block_1(h, temb)
            h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
            #     if len(self.up[i_level].attn) > 0:
            #         h = self.up[i_level].attn[i_block](h)
            # if i_level != 0:
            #     h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        if self.use_act:
            h = nonlinearity(h)
        h = self.conv_out(h)
        return h


   

