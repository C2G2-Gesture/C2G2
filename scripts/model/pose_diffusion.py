import torch
import torch.nn as nn

from .diffusion_net import *
from .diffusion_util import *
from .diffusion_ae import *
from .motion_ae import *
from .latent_diffusion import *
from .vqvae import *

class PoseDiffusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pre_length = args.n_pre_poses
        self.gen_length = args.n_poses - args.n_pre_poses
        self.input_context = args.input_context
        self.audio_decoder = False

        # add attribute args for sampling
        self.args = args
        pose_dim = args.pose_dim
        diff_hidden_dim = args.diff_hidden_dim
        block_depth = args.block_depth

        self.in_size = 32 + pose_dim + 1 
        self.audio_encoder = WavEncoder()
        if self.audio_decoder:
            self.audio_decoder = WavDecoder()

        self.classifier_free = args.classifier_free
        if self.classifier_free:
            self.null_cond_prob = args.null_cond_prob
            self.null_cond_emb = nn.Parameter(torch.randn(1, self.in_size))

        self.diffusion_net = DiffusionNet(
            net = TransformerModel( num_pose=args.n_poses,
                                    pose_dim=pose_dim, 
                                    embed_dim=pose_dim+3+self.in_size,
                                    hidden_dim=diff_hidden_dim,
                                    depth=block_depth//2,
                                    decoder_depth=block_depth//2
                                    ),
            var_sched = VarianceSchedule(
                num_steps=500,
                beta_1=1e-4,
                beta_T=0.02,
                mode='linear'
            )
        )

    def get_loss(self, x, pre_seq, in_audio,in_mel):
        if self.input_context == 'audio':
            if in_mel!=None:
                audio_feat_seq = self.audio_encoder(in_audio,in_mel)
            else:
                audio_feat_seq = self.audio_encoder(in_audio) # output (bs, n_frames, feat_size)
            in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
        else:
            assert False


        if self.classifier_free:
            mask = torch.zeros((x.shape[0],), device = x.device).float().uniform_(0, 1) < self.null_cond_prob
            in_data = torch.where(mask.unsqueeze(1).unsqueeze(2), self.null_cond_emb.repeat(in_data.shape[1],1).unsqueeze(0), in_data)
        
        if self.audio_decoder:
            recon_audio = self.audio_decoder(audio_feat_seq)
            loss = nn.MSELoss()
            recon_loss = loss(recon_audio,in_audio).mean()
        else:
            recon_loss = torch.Tensor([0.0]).to(x.device)

        neg_elbo = self.diffusion_net.get_loss(x, in_data)

        return neg_elbo, recon_loss
        
    def sample(self, pose_dim, pre_seq, in_audio):

        if self.input_context == 'audio':
            audio_feat_seq = self.audio_encoder(in_audio)
            in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)

        if self.classifier_free:
            uncondition_embedding = self.null_cond_emb.repeat(in_data.shape[1],1).unsqueeze(0)
            samples = self.diffusion_net.sample(self.args.n_poses, in_data, pose_dim, uncondition_embedding=uncondition_embedding)
        else:
            samples = self.diffusion_net.sample(self.args.n_poses, in_data, pose_dim)
        return samples

class PoseLatentDiffusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pre_length = args.n_pre_poses
        self.gen_length = args.n_poses - args.n_pre_poses
        self.input_context = args.input_context
        self.pred_real = True
        self.device= args.device

        # add attribute args for sampling
        self.args = args
        pose_dim = args.pose_dim
        
        diff_hidden_dim = args.diff_hidden_dim
        block_depth = args.block_depth
        

        
        if pose_dim == 27:
            self.vqvae = VQModel(in_channels=27,n_embed=1024,embed_dim=64,hidden_channels=64,num_res_blocks=2,device=self.device)
            self.latent_dim = 64
        elif pose_dim == 126:
            self.vqvae = VQModel(in_channels=126,n_embed=1024,embed_dim=128,hidden_channels=128,num_res_blocks=2,device=self.device)
            self.latent_dim = 128
        else:
            assert False
            
        self.in_size = 32 + self.latent_dim
        # self.audio_encoder = WavEncoder()
        
        self.combine_mel = False
        
        if self.combine_mel:
            self.audio_encoder = BeatWavEncoder_combine()
        else:
            self.audio_encoder = WavEncoder()
            
        self.use_decoder = False

        if self.use_decoder:
            self.audio_decoder = WavDecoder()
        self.vqvae_weight = args.vqvae_weight

        self.classifier_free = args.classifier_free
        if self.classifier_free:
            self.null_cond_prob = args.null_cond_prob
            self.null_cond_emb = nn.Parameter(torch.randn(1, self.in_size))

        self.diffusion_net = LatentDiffusionNet(
            net = TransformerModel(num_pose=args.n_poses,
                                    pose_dim=self.latent_dim, 
                                    embed_dim=self.latent_dim+3+self.in_size,
                                    hidden_dim=diff_hidden_dim,
                                    depth=block_depth//2,
                                    decoder_depth=block_depth//2
                                    ),
            var_sched = VarianceSchedule(
                num_steps=500,
                beta_1=1e-4,
                beta_T=0.02,
                mode='linear'
            ),
            VQ=self.vqvae
        )
        self.encoder = self.vqvae.encoder
        self.load_vae()
        
    def load_vae(self):
        vae_weight = torch.load(self.vqvae_weight,map_location="cpu")["state_dict"]
        # load vae used in diffusion
        self.diffusion_net.vqvae.load_state_dict(vae_weight,strict=True)
        # load pre seq encoder
        self.encoder.load_state_dict(vae_weight,strict=False)
        self.encoder = self.encoder.eval()
        self.encoder.train = disabled_train
        for param in self.encoder.parameters():
            param.requires_grad = False
        

    def get_loss(self, x, pre_seq, in_audio, in_mel=None,clip_pre=False):

        if self.input_context == 'audio':
            if self.combine_mel:
                assert in_mel!=None
                audio_feat_seq = self.audio_encoder(in_audio,in_mel)
            else:
                audio_feat_seq = self.audio_encoder(in_audio) # output (bs, n_frames, feat_size)
            with torch.no_grad():
                if clip_pre:
                    pre_seq = pre_seq[:,:self.pre_length,:-1]
                    pre_seq = self.encoder(pre_seq)
                    # pre_seq = pre_seq.transpose(1,2)
                    # pre_seq = self.vqvae.quant_conv(pre_seq).transpose(1,2)
                    pre_seq_padding = torch.zeros((pre_seq.shape[0],self.gen_length,pre_seq.shape[2])).to(pre_seq.device)
                    pre_seq = torch.cat((pre_seq,pre_seq_padding),1)
                else:
                    pre_seq = self.encoder(pre_seq[:,:,:-1])
            in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
        else:
            assert False
        

        if self.classifier_free:
            mask = torch.zeros((x.shape[0],), device = x.device).float().uniform_(0, 1) < self.null_cond_prob
            in_data = torch.where(mask.unsqueeze(1).unsqueeze(2), self.null_cond_emb.repeat(in_data.shape[1],1).unsqueeze(0), in_data)
            
        if self.use_decoder:
            recon_audio = self.audio_decoder(audio_feat_seq)
            loss = nn.MSELoss()
            recon_loss = loss(recon_audio,in_audio).mean()
        else:
            recon_loss = torch.Tensor([0.0]).to(x.device)

        neg_elbo = self.diffusion_net.get_loss(x, in_data)

        return neg_elbo, recon_loss
        
    def sample(self, pose_dim, pre_seq, in_audio, audio_feature=None, target_vec=None, mask=None, use_repaint=False):
        if pose_dim == 27:
            latent_dim = 64
        else:
            latent_dim = 128
            

        if self.input_context == 'audio':
            if self.combine_mel:
                assert audio_feature!=None
                audio_feat_seq = self.audio_encoder(in_audio,audio_feature)
            else:
                audio_feat_seq = self.audio_encoder(in_audio)
            with torch.no_grad():
                pre_seq = self.encoder(pre_seq[:,:,:-1])
            in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
            
        if use_repaint:
            if self.classifier_free:
                uncondition_embedding = self.null_cond_emb.repeat(in_data.shape[1],1).unsqueeze(0)
                print(mask.shape)
                samples,latent = self.diffusion_net.repaint_latent_sample(self.args.n_poses, in_data, latent_dim, target_vec, mask, uncondition_embedding=uncondition_embedding)
            else:
                samples,latent = self.diffusion_net.repaint_latent_sample(self.args.n_poses, in_data, latent_dim, target_vec, mask)
        else:
            if self.classifier_free:
                uncondition_embedding = self.null_cond_emb.repeat(in_data.shape[1],1).unsqueeze(0)
                samples,latent = self.diffusion_net.latent_sample(self.args.n_poses, in_data, latent_dim, uncondition_embedding=uncondition_embedding)
            else:
                samples,latent = self.diffusion_net.latent_sample(self.args.n_poses, in_data, latent_dim)
                
       
        return samples, latent
    
    def ddim_sample(self, pose_dim, pre_seq, in_audio, ddim_timesteps = 50, target_vec=None, mask=None, use_repaint=False):
        if pose_dim == 27:
            latent_dim = 64
        else:
            latent_dim = 128

        if self.input_context == 'audio':
            audio_feat_seq = self.audio_encoder(in_audio)
            with torch.no_grad():
                pre_seq = self.encoder(pre_seq[:,:,:-1])
            in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
            
        if self.classifier_free:
            uncondition_embedding = self.null_cond_emb.repeat(in_data.shape[1],1).unsqueeze(0)
            samples,latent = self.diffusion_net.ddim_latent_sample(self.args.n_poses, in_data, latent_dim, ddim_timesteps, uncondition_embedding=uncondition_embedding)
        else:
            samples,latent = self.diffusion_net.ddim_latent_sample(self.args.n_poses, in_data, latent_dim, ddim_timesteps)
        
       
        return samples, latent

class WavEncoder(nn.Module):
    def __init__(self, out_dim = None):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=5, padding=1600), #36267+3200-15/5+1 7891
            nn.BatchNorm1d(16), 
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, 32, 15, stride=6), #7891-15/6 +1 =1314 
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(32, 64, 15, stride=6), #1314-15/6+1 = 218
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(64, 32, 15, stride=6), #218-15/6+1 34
        )
        if out_dim!=None:
            self.linear = nn.Linear(34,out_dim)
        self.out_dim = out_dim

    def forward(self, wav_data):
        wav_data = wav_data.unsqueeze(1)  # add channel dim
        out = self.feat_extractor(wav_data)
        out = out.transpose(1, 2)
        if self.out_dim!=None:
            out = self.linear(out)

        return  out# to (batch x seq x dim)
    
class WavEncoder_trans(nn.Module):
    def __init__(self, out_dim = None):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=5, padding=1600), #36267+3200-15/5+1 7891
            nn.BatchNorm1d(16), 
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, 32, 15, stride=6), #7891-15/6 +1 =1314 
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(32, 64, 15, stride=6), #1314-15/6+1 = 218
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(64, 32, 15, stride=6), #218-15/6+1 34
        )
        self.transencoder = TransformerEncoder(num_pose=34, pose_dim=32, depth=2, num_heads=4)
        if out_dim!=None:
            self.linear = nn.Linear(34,out_dim)
        self.out_dim = out_dim

    def forward(self, wav_data):
        wav_data = wav_data.unsqueeze(1)  # add channel dim
        out = self.feat_extractor(wav_data)
        out = out.transpose(1, 2)
        if self.out_dim!=None:
            out = self.linear(out)
        out = self.transencoder(out)

        return  out# to (batch x seq x dim)
    
class WavEncoder_v2(nn.Module):
    def __init__(self, out_dim = 34):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(23, 32, 3, stride=1,padding=1), #36267+3200-15/5+1 7891
            nn.BatchNorm1d(32), 
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(32, 64, 3, stride=1,padding=1), #7891-15/6 +1 =1314 
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(64, 32, 3, stride=1,padding=1), #1314-15/6+1 = 218
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(32, 32, 3,1,1), #218-15/6+1 34
        )
        
        if out_dim!=None:
            self.linear = nn.Sequential(
                nn.Linear(71,out_dim),
                nn.LayerNorm(out_dim),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Linear(out_dim,out_dim),
                
            )
        self.out_dim = out_dim

    def forward(self, wav_data):
        wav_data = wav_data.transpose(1,2)
        out = self.feat_extractor(wav_data)
        if self.out_dim!=None:
            out = self.linear(out)
        out = out.transpose(1,2)

        return  out# to (batch x seq x dim)
    
class WavEncoder_lstm(nn.Module):
    def __init__(self, out_dim = None):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=5, padding=1600), #36267+3200-15/5+1 7891
            nn.BatchNorm1d(16), 
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, 32, 15, stride=6), #7891-15/6 +1 =1314 
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(32, 64, 15, stride=6), #1314-15/6+1 = 218
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(64, 32, 15, stride=6), #218-15/6+1 34
        )
        self.lstm_layer = nn.LSTM(input_size=32, hidden_size=32, num_layers=2,bidirectional=True)
 
        self.linear = nn.Linear(64,32)
        self.out_dim = out_dim

    def forward(self, wav_data):
        wav_data = wav_data.unsqueeze(1)  # add channel dim
        out = self.feat_extractor(wav_data)
        out = out.transpose(1, 2)
        out,_ = self.lstm_layer(out)
        out = self.linear(out)

        return  out# to (batch x seq x dim)
    
class WavEncoder_v3(nn.Module):
    def __init__(self, out_dim = 32):
        super().__init__()
        self.audio_encoder = WavEncoder()
        self.feature_encoder = WavEncoder_v2()
        self.transencoder = TransformerEncoder(num_pose=34,
                 pose_dim=out_dim*2, depth=2, num_heads=4)
        
        self.combine = nn.Sequential(
            nn.Linear(out_dim*2,out_dim),
            nn.LeakyReLU(0.2,True),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim,out_dim),
            nn.LeakyReLU(0.2,True),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim,out_dim),
        )

    def forward(self, wav_data,feature_data):
        feature_enc = self.audio_encoder(wav_data)
        audio_enc = self.feature_encoder(feature_data)
        out = torch.cat((feature_enc,audio_enc),2)
        out = self.transencoder(out)
        out = self.combine(out)
        return  out# to (batch x seq x dim)
    
class BeatEncoder(nn.Module):
    def __init__(self, out_dim = 34):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(3, 16, 3, stride=1,padding=1), #36267+3200-15/5+1 7891
            nn.BatchNorm1d(16), 
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, 16, 3, stride=1,padding=1), #7891-15/6 +1 =1314 
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, 16, 3, stride=1,padding=1), #1314-15/6+1 = 218
        )
        
        self.linear = nn.Sequential(
            nn.Linear(71,out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Linear(out_dim,out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Linear(out_dim,out_dim),
        )
        
        
        self.out_dim = out_dim

    def forward(self, wav_data):
        wav_data = wav_data.transpose(1,2)
        out = self.feat_extractor(wav_data)
        if self.out_dim!=None:
            out = self.linear(out)
        out = out.transpose(1, 2)

        return  out# to (batch x seq x dim)
    
class BeatWavEncoder_combine(nn.Module):
    def __init__(self, out_dim = 32,input_dim=32+16):
        super().__init__()
        self.audio_encoder = WavEncoder()
        self.beat_encoder = BeatEncoder()
        self.transencoder = TransformerEncoder(num_pose=34,
                 pose_dim=input_dim, depth=2, num_heads=4)
        self.combine = nn.Sequential(
            nn.Linear(input_dim,out_dim),
            nn.LeakyReLU(0.2,True),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim,out_dim),
            nn.LeakyReLU(0.2,True),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim,out_dim),
        )

    def forward(self, wav_data,feature_data):
        feature_enc = self.audio_encoder(wav_data)
        audio_enc = self.beat_encoder(feature_data)
        out = torch.cat((feature_enc,audio_enc),2)
        out = self.transencoder(out)
        out = self.combine(out)
        return  out# to (batch x seq x dim)

class WavDecoder(nn.Module):
    def __init__(self, out_dim = None):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.ConvTranspose1d(32, 32, 15, stride=6), #33*6+15 = 213
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3, inplace=True),
            nn.ConvTranspose1d(32, 16, 15, stride=6), #1287
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.ConvTranspose1d(16, 16, 15, stride=6), #7731
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.ConvTranspose1d(16, 1, 15, stride=5,padding=1199), 
        )
        if out_dim!=None:
            self.linear = nn.Linear(out_dim,34)
        self.out_dim = out_dim

    def forward(self, wav_feat):
        # wav_data = wav_data.unsqueeze(1)  # add channel dim
        wav_feat = wav_feat.transpose(1,2)
        if self.out_dim !=None:
            wav_feat = self.linear(wav_feat)
        out = self.feat_extractor(wav_feat)
        return out.transpose(1, 2).squeeze(2)  # to (batch x seq x dim)
