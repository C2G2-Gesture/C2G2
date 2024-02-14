from torch.utils.data import DataLoader
import datetime
import librosa
import lmdb
import logging
import math
import numpy as np
import os
import pickle
import pprint
import pyarrow
import random
import soundfile as sf
import sys
import time
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize

from data_loader.data_preprocessor import DataPreprocessor
from data_loader.lmdb_data_loader_expressive import SpeechMotionDataset, default_collate_fn
from model.embedding_space_evaluator import EmbeddingSpaceEvaluator
from utils.average_meter import AverageMeter
from utils.data_utils_expressive import convert_dir_vec_to_pose, convert_pose_seq_to_dir_vec, resample_pose_seq, dir_vec_pairs, convert_dir_vec_to_pose_fixed, convert_dir_vec_to_pose_l2n, convert_dir_vec_to_norm_vec
from utils.train_utils import set_logger, set_random_seed
from model.vqvae import VQModel,VAEModel, CondVQModel

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

angle_pair = [
    (0, 1),
    (0, 2),
    (1, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (8, 9),
    (9, 10),
    (11, 12),
    (12, 13),
    (14, 15),
    (15, 16),
    (17, 18),
    (18, 19),
    (17, 5),
    (5, 8),
    (8, 14),
    (14, 11),
    (2, 20),
    (20, 21),
    (22, 23),
    (23, 24),
    (25, 26),
    (26, 27),
    (28, 29),
    (29, 30),
    (31, 32),
    (32, 33),
    (34, 35),
    (35, 36),
    (34, 22),
    (22, 25),
    (25, 31),
    (31, 28),
    (0, 37),
    (37, 38),
    (37, 39),
    (38, 40),
    (39, 41),
    # palm
    (4, 42),
    (21, 43)
]

change_angle = [0.0027804733254015446, 0.002761547453701496, 0.005953566171228886, 0.013764726929366589, 
    0.022748252376914024, 0.039307352155447006, 0.03733552247285843, 0.03775784373283386, 0.0485558956861496, 
    0.032914578914642334, 0.03800227493047714, 0.03757007420063019, 0.027338404208421707, 0.01640886254608631, 
    0.003166505601257086, 0.0017252820543944836, 0.0018696568440645933, 0.0016072227153927088, 0.005681346170604229, 
    0.013287615962326527, 0.021516695618629456, 0.033936675637960434, 0.03094293735921383, 0.03378918394446373, 
    0.044323261827230453, 0.034706637263298035, 0.03369896858930588, 0.03573163226246834, 0.02628341130912304, 
    0.014071882702410221, 0.0029828345868736506, 0.0015706412959843874, 0.0017107439925894141, 0.0014634154504165053, 
    0.004873405676335096, 0.002998138777911663, 0.0030240598134696484, 0.0009890805231407285, 0.0012279648799449205, 
    0.047324635088443756, 0.04472292214632034]

sigma = 0.1
thres = 0.001

from model.pose_diffusion import PoseDiffusion, PoseLatentDiffusion

def load_checkpoint_and_model(checkpoint_path, _device='cpu'):
    print('loading checkpoint {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    args = checkpoint['args']
    epoch = checkpoint['epoch']
    lang_model = checkpoint['lang_model']
    speaker_model = checkpoint['speaker_model']
    pose_dim = checkpoint['pose_dim']
    print('epoch {}'.format(epoch))

    print("init diffusion model")
    diffusion = PoseDiffusion(args).to(_device)

    diffusion.load_state_dict(checkpoint['state_dict'])

    return args, diffusion, lang_model, speaker_model, pose_dim

def load_checkpoint_and_latent_model(checkpoint_path, _device='cpu'):
    print('loading checkpoint {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    args = checkpoint['args']

    args.vqvae_weight = '/data/longbinji/checkpoints_c2g2_best/expressive/train_vqvae_expressive_v2/pose_diffusion_checkpoint_300.bin'
    args.output_path = '/data/longbinji/checkpoints_c2g2_best/expressive/paper_train_latent_300_lower_lr'
    
    epoch = checkpoint['epoch']
    lang_model = checkpoint['lang_model']
    speaker_model = checkpoint['speaker_model']
    pose_dim = checkpoint['pose_dim']
    print('epoch {}'.format(epoch))

    print("init diffusion model")
    args.device = _device
    diffusion = PoseLatentDiffusion(args).to(_device)
    if combine_mel:
        diffusion.combine_mel = True

    diffusion.load_state_dict(checkpoint['state_dict'])
        

    return args, diffusion, lang_model, speaker_model, pose_dim

def load_checkpoint_and_vqmodel(checkpoint_path, _device='cpu'):
    print('loading checkpoint {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    args = checkpoint['args']
    epoch = checkpoint['epoch']
    lang_model = checkpoint['lang_model']
    speaker_model = checkpoint['speaker_model']
    pose_dim = checkpoint['pose_dim']
    print('epoch {}'.format(epoch))

    print("init diffusion model")
    if mode_model=="vqvae":
        if True:
            vqvae = VQModel(in_channels=126,n_embed=1024,embed_dim=128,hidden_channels=128,num_res_blocks=2).to(_device)
        else:
            vqvae = CondVQModel(in_channels=126,n_embed=1024,embed_dim=128,hidden_channels=128,num_res_blocks=2).to(_device)

    vqvae.load_state_dict(checkpoint['state_dict'],strict=True)
    # vqvae.eval()

    return args, vqvae, lang_model, speaker_model, pose_dim


def generate_gestures(args, diffusion, lang_model, audio, words, pose_dim, audio_sr=16000,
                      seed_seq=None, fade_out=False, target_vec=None, last_resample=False, pred_real=True, real_dir_vec=None, middle_resample=False):
    out_list = []
    
    if condition_mode:
        real_out_list = []
        vqcond = CondVQModel(in_channels=126,n_embed=1024,embed_dim=128,hidden_channels=128,num_res_blocks=2).to(device)
        weight = torch.load('/home/tiger/nfs/workspace/DiffGesture/output/paper_vqvae_cond_300/pose_diffusion_checkpoint_350.bin')["state_dict"]
        vqcond.load_state_dict(weight,strict=True)
        
        # prepare length information using first frame
        real_dir_vec = torch.from_numpy(real_dir_vec).to(device).float()
        real_dir_vec = real_dir_vec.unsqueeze(0)
        
        print(real_dir_vec.shape)
        
    n_frames = args.n_poses
    clip_length = len(audio) / audio_sr

    # pre seq
    pre_seq = torch.zeros((1, n_frames, len(args.mean_dir_vec) + 1))
    if seed_seq is not None:
        pre_seq[0, 0:args.n_pre_poses, :-1] = torch.Tensor(seed_seq[0:args.n_pre_poses])
        pre_seq[0, 0:args.n_pre_poses, -1] = 1  # indicating bit for seed poses

    # divide into synthesize units and do synthesize
    unit_time = args.n_poses / args.motion_resampling_framerate
    stride_time = (args.n_poses - args.n_pre_poses) / args.motion_resampling_framerate
    if clip_length < unit_time:
        num_subdivision = 1
    else:
        num_subdivision = math.ceil((clip_length - unit_time) / stride_time) + 1
    audio_sample_length = int(unit_time * audio_sr)
    end_padding_duration = 0

    print('{}, {}, {}, {}, {}'.format(num_subdivision, unit_time, clip_length, stride_time, audio_sample_length))

    out_dir_vec = None
    start = time.time()
    
    if last_resample:
        assert len(target_vec)>0
        
    for i in range(0, num_subdivision):
        start_time = i * stride_time
        end_time = start_time + unit_time
        
        unit_length = 30
        start_frame = i*unit_length
        end_frame = (i+1)*unit_length

        # prepare audio input
        audio_start = math.floor(start_time / clip_length * len(audio))
        audio_end = audio_start + audio_sample_length
        in_audio = audio[audio_start:audio_end]
        if len(in_audio) < audio_sample_length:
            print(f"error: start {start_frame}, end {end_frame}")
            if i == num_subdivision - 1:
                end_padding_duration = audio_sample_length - len(in_audio)
            in_audio = np.pad(in_audio, (0, audio_sample_length - len(in_audio)), 'constant')
        in_audio = torch.from_numpy(in_audio).unsqueeze(0).to(device).float()

        # prepare text input
        word_seq = DataPreprocessor.get_words_in_time_range(word_list=words, start_time=start_time, end_time=end_time)
        extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
        word_indices = np.zeros(len(word_seq) + 2)
        word_indices[0] = lang_model.SOS_token
        word_indices[-1] = lang_model.EOS_token
        frame_duration = (end_time - start_time) / n_frames
        for w_i, word in enumerate(word_seq):
            print(word[0], end=', ')
            idx = max(0, int(np.floor((word[1] - start_time) / frame_duration)))
            extended_word_indices[idx] = lang_model.get_word_index(word[0])
            word_indices[w_i + 1] = lang_model.get_word_index(word[0])
        print(' ')

        # prepare pre seq
        if i > 0:
            pre_seq[0, 0:args.n_pre_poses, :-1] = out_dir_vec.squeeze(0)[-args.n_pre_poses:]
            pre_seq[0, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq = pre_seq.float().to(device)
        
        if last_resample:
            print("use last resample")
            if i == num_subdivision - 1:
                out_seq = target_vec[start_frame:,:]
                
            elif i == num_subdivision - 2:

                chunk_target_vec = target_vec[int(start_frame):int(end_frame+args.n_pre_poses)]
                
                mask = torch.zeros((unit_length+args.n_pre_poses,128))
                mask_start = int(0.2*unit_length)
                chunk_target_vec = chunk_target_vec[-mask_start:,:]
                print(f"masked length {mask_start}")
                mask[-mask_start:,:] = 1.0
                mask = mask.to(device).float()
                chunk_target_vec = torch.from_numpy(chunk_target_vec).to(device).float()
                if args.model == 'pose_diffusion':
                    out_dir_vec,latent = diffusion.sample(pose_dim, pre_seq, in_audio,target_vec=chunk_target_vec,mask = mask,use_repaint=True)
                    
                out_seq = out_dir_vec[0, :, :].data.cpu().numpy()
            else:
                if args.model == 'pose_diffusion':
                    out_dir_vec,latent = diffusion.sample(pose_dim, pre_seq, in_audio)
                    
                # with torch.no_grad():
                #     out_dir_vec = vqvae.quantize(latent)
                #     out_dir_vec = vqvae.decode(out_dir_vec)

                out_seq = out_dir_vec[0, :, :].data.cpu().numpy()   
                   
        elif middle_resample:
            print("use middle resample")
                
            if i == 2:
                chunk_target_vec = target_vec[int(start_frame):int(end_frame+args.n_pre_poses)]
                
                
                mask = torch.zeros((unit_length+args.n_pre_poses,128))
                print(int(end_frame+args.n_pre_poses)-24)
                mask_start = 24
                chunk_target_vec = chunk_target_vec[-mask_start:,:]
                print(f"masked length {mask_start}")
                mask[-mask_start:,:] = 1.0
                mask = mask.to(device).float()
                chunk_target_vec = torch.from_numpy(chunk_target_vec).to(device).float()
                if args.model == 'pose_diffusion':
                    out_dir_vec,latent = diffusion.sample(pose_dim, pre_seq, in_audio,target_vec=chunk_target_vec,mask = mask,use_repaint=True)
                    
                out_seq = out_dir_vec[0, :, :].data.cpu().numpy()
            else:
                if args.model == 'pose_diffusion':
                    out_dir_vec,latent = diffusion.sample(pose_dim, pre_seq, in_audio)
                    
                # with torch.no_grad():
                #     out_dir_vec = vqvae.quantize(latent)
                #     out_dir_vec = vqvae.decode(out_dir_vec)

                out_seq = out_dir_vec[0, :, :].data.cpu().numpy()  
        else:
            if args.model == 'pose_diffusion':
                if mode_model == "latent_diffusion":
                    out_dir_vec,latent = diffusion.sample(pose_dim, pre_seq, in_audio)
                    out_seq = out_dir_vec[0, :, :].data.cpu().numpy()
                else:
                    if i == num_subdivision - 1:
                        out_seq = target_vec[start_frame:,:]
                    else:
                        out_dir_vec = diffusion.sample(pose_dim, pre_seq, in_audio)
                        out_seq = out_dir_vec[0, :, :].data.cpu().numpy()
                        # if i == num_subdivision - 2:
                        #     out_seq[-6:,:] = target_vec[-6:,:]

        # smoothing motion transition
        if len(out_list) > 0:
            last_poses = out_list[-1][-args.n_pre_poses:]
            out_list[-1] = out_list[-1][:-args.n_pre_poses]  # delete last 4 frames

            for j in range(len(last_poses)):
                n = len(last_poses)
                prev = last_poses[j]
                next = out_seq[j]
                out_seq[j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)
        
        # predict real length vectors
        if condition_mode:
            print("start to predict real")
            if i == num_subdivision - 1:
                if last_resample:
                    real_out_seq = real_dir_vec[0,start_frame:,:]
                    real_out_seq = real_out_seq.data.cpu().numpy()
                else:
                    with torch.no_grad():
                        quant,_,_ = vqcond.quantize(latent)
                        real_out_vec = vqcond.decode(quant,real_dir_vec[:,0,:])
                        real_out_seq = real_out_vec[0, :, :].data.cpu().numpy()
            else:
                with torch.no_grad():
                    quant,_,_ = vqcond.quantize(latent)
                    real_out_vec = vqcond.decode(quant,real_dir_vec[:,0,:])
                    real_out_seq = real_out_vec[0, :, :].data.cpu().numpy()
        
            
            # smoothing motion transition
            if len(real_out_list) > 0:
                last_poses = real_out_list[-1][-args.n_pre_poses:]
                real_out_list[-1] = real_out_list[-1][:-args.n_pre_poses]  # delete last 4 frames

                for j in range(len(last_poses)):
                    n = len(last_poses)
                    prev = last_poses[j]
                    next = real_out_seq[j]
                    real_out_seq[j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)
            real_out_list.append(real_out_seq)

        out_list.append(out_seq)
        

    print('generation took {:.2} s'.format((time.time() - start) / num_subdivision))

    # aggregate results
    if condition_mode:
        out_dir_vec = np.vstack(real_out_list)
    else:
        out_dir_vec = np.vstack(out_list)

    # fade out to the mean pose
    if fade_out:
        n_smooth = args.n_pre_poses
        start_frame = len(out_dir_vec) - int(end_padding_duration / audio_sr * args.motion_resampling_framerate)
        end_frame = start_frame + n_smooth * 2
        if len(out_dir_vec) < end_frame:
            out_dir_vec = np.pad(out_dir_vec, [(0, end_frame - len(out_dir_vec)), (0, 0)], mode='constant')
        out_dir_vec[end_frame-n_smooth:] = np.zeros((len(args.mean_dir_vec)))  # fade out to mean poses

        # interpolation
        y = out_dir_vec[start_frame:end_frame]
        x = np.array(range(0, y.shape[0]))
        w = np.ones(len(y))
        w[0] = 5
        w[-1] = 5
        coeffs = np.polyfit(x, y, 2, w=w)
        fit_functions = [np.poly1d(coeffs[:, k]) for k in range(0, y.shape[1])]
        interpolated_y = [fit_functions[k](x) for k in range(0, y.shape[1])]
        interpolated_y = np.transpose(np.asarray(interpolated_y))  # (num_frames x dims)

        out_dir_vec[start_frame:end_frame] = interpolated_y

    return out_dir_vec




def evaluate_testset(test_data_loader, diffusion, embed_space_evaluator, args, pose_dim, only_FGD=True):

    if embed_space_evaluator:
        embed_space_evaluator.reset()
    # losses = AverageMeter('loss')
    joint_mae = AverageMeter('mae_on_joint')
    accel = AverageMeter('accel')
    bc = AverageMeter('bc')
    start = time.time()

    
    if condition_mode:
        vqcond = CondVQModel(in_channels=126,n_embed=1024,embed_dim=128,hidden_channels=128,num_res_blocks=2).to(device)
        weight = torch.load('/data/longbinji/checkpoints_c2g2_best/expressive/paper_vqvae_cond_300/pose_diffusion_checkpoint_499.bin')["state_dict"]
        # weight = torch.load('/home/tiger/nfs/workspace/DiffGesture/output/fixed_paper_train_vqvae_expressive_cond_best/pose_diffusion_checkpoint_300.bin')["state_dict"]
        vqcond.load_state_dict(weight,strict=True)
        

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            print("testing {}/{}".format(iter_idx, len(test_data_loader)))
            if condition_mode or direct_rl:
                in_text, _, in_text_padded, _, target_vec, in_audio, in_spec, _,real_dir_vec = data
            else:
                in_text, _, in_text_padded, _, target_vec, in_audio, in_spec, _ = data
            in_audio = in_audio.detach().cpu().numpy()
            if combine_mel:
                audio_features = []
                for audio in in_audio:
                    FPS = 60
                    HOP_LENGTH = 512
                    SR = FPS * HOP_LENGTH
                    mfcc = librosa.feature.mfcc(audio, sr=SR, n_mfcc=20).T
                    envelope = librosa.onset.onset_strength(audio, sr=SR)
                    peak_idxs = librosa.onset.onset_detect(
                        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH)
                    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
                    peak_onehot[peak_idxs] = 1.0  # (seq_len,)

                    tempo, beat_idxs = librosa.beat.beat_track(
                        onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH,
                        tightness=100)
                    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
                    beat_onehot[beat_idxs] = 1.0  # (seq_len,)
                    # audio_feature = np.concatenate([
                    #     envelope[:, None], mfcc,peak_onehot[:, None], beat_onehot[:, None]
                    #         ], axis=-1)
                    audio_feature = np.concatenate([
                        envelope[:, None], peak_onehot[:, None], beat_onehot[:, None]
                            ], axis=-1)
                    audio_features.append(audio_feature)
                audio_features = np.stack(audio_features,0)
            
            batch_size = target_vec.size(0)

            in_text = in_text.to(device)
            in_text_padded = in_text_padded.to(device)

            in_spec = in_spec.to(device)
            target = target_vec.to(device)

            if condition_mode or direct_rl:
                real_dir_vec = real_dir_vec.to(device)
            

            pre_seq = target.new_zeros((target.shape[0], target.shape[1], target.shape[2] + 1))
            pre_seq[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses]
            pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
            
            if direct_rl:
                in_audio = torch.from_numpy(in_audio).to(device)
                pre_seq_rl = real_dir_vec.new_zeros((real_dir_vec.shape[0], real_dir_vec.shape[1], real_dir_vec.shape[2] + 1))
                pre_seq_rl[:, 0:args.n_pre_poses, :-1] = real_dir_vec[:, 0:args.n_pre_poses]
                pre_seq_rl[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
                real_out_vec = diffusion.sample(pose_dim, pre_seq_rl, in_audio)

            elif args.model == 'pose_diffusion':
                if mode_model =="vqvae":
                    print("generate for vqvae")
                    in_audio = torch.from_numpy(in_audio).to(device)
                    if condition_mode:
                        # real_out_vec,_ = diffusion(target,real_dir_vec[:,0,:])
                        # out_dir_vec,latent = diffusion(target)
                        h = diffusion.encoder(target)
                        h = h.transpose(1,2)
                        latent = diffusion.quant_conv(h).transpose(1,2)
                        print("start to predict real")
                        with torch.no_grad():
                            quant,_,_ = vqcond.quantize(latent)
                            real_out_vec = vqcond.decode(quant,real_dir_vec[:,0,:])
                    else:
                        out_dir_vec,_ = diffusion(target)
                else: 
                    if combine_mel:
                        audio_features = torch.from_numpy(audio_features).to(device)
                        in_audio = torch.from_numpy(in_audio).to(device)
                        # out_dir_vec,_ = diffusion.sample(pose_dim, pre_seq, audio_features)
                        # out_dir_vec,_ = diffusion.ddim_sample(pose_dim, pre_seq,in_audio, audio_feature=audio_features,ddim_timesteps=50)
                        out_dir_vec,latent = diffusion.sample(pose_dim, pre_seq,in_audio, audio_feature=audio_features)
                    else:
                        in_audio = torch.from_numpy(in_audio).to(device)
                        # out_dir_vec,latent = diffusion.sample(pose_dim, pre_seq, in_audio)

                        if repainting_only:
                            # chunk_target_vec = target_dir_vec[select_index,:,:]
                            mask = torch.zeros((target.shape[1],128))
                            mask_start = 6
                            mask[-mask_start:,:] = 1.0
                            mask = mask.to(device).float()
                            # chunk_target_vec = chunk_target_vec.to(device)
                            # in_audio = torch.from_numpy(in_audio).to(device)
                            # in_audio = torch.zeros(in_audio.shape).to(in_audio.device)
                            print(f"masked repainting with length {mask_start}")
                            out_dir_vec,latent = diffusion.sample(pose_dim, pre_seq, in_audio, target_vec=target,mask = mask,use_repaint=True)
                            # out_dir_vec[:,-2:,:] = target_dir_vec[:,-2:,:]
                        else:
                            out_dir_vec,latent = diffusion.sample(pose_dim, pre_seq, in_audio)
                        if condition_mode:
                            print("start to predict real")
                            with torch.no_grad():
                                quant,_,_ = vqcond.quantize(latent)
                                real_out_vec = vqcond.decode(quant,real_dir_vec[:,0,:])
                    
            if not (condition_mode or direct_rl):
                out_dir_vec_bc = out_dir_vec + torch.tensor(args.mean_dir_vec).squeeze(1).unsqueeze(0).unsqueeze(0).to(device)
            else:
                print("real length mode")
                out_dir_vec_bc = real_out_vec
                
            if only_FGD:
                if args.model != 'gesture_autoencoder':
                    if embed_space_evaluator:
                        if not (condition_mode or direct_rl):
                            embed_space_evaluator.push_samples(in_text_padded, in_audio, out_dir_vec, target)
                        else:
                            real_out_vec_eval = real_out_vec.clone().cpu().numpy()
                            real_dir_vec_eval = real_dir_vec.clone().cpu().numpy()
                            unit_real_out = torch.from_numpy(convert_dir_vec_to_norm_vec(real_out_vec_eval)).to(device)
                            unit_dir_vec = torch.from_numpy(convert_dir_vec_to_norm_vec(real_dir_vec_eval)).to(device)
                            unit_real_out = unit_real_out.reshape(unit_real_out.shape[0],unit_real_out.shape[1],-1)
                            unit_dir_vec = unit_dir_vec.reshape(unit_real_out.shape[0],unit_real_out.shape[1],-1)
                            print(unit_dir_vec.shape)
                
                            embed_space_evaluator.push_samples(in_text_padded, in_audio, unit_real_out,unit_dir_vec)
            else:
            
                left_palm = torch.cross(out_dir_vec_bc[:, :, 11 * 3 : 12 * 3], out_dir_vec_bc[:, :, 17 * 3 : 18 * 3], dim = 2)
                right_palm = torch.cross(out_dir_vec_bc[:, :, 28 * 3 : 29 * 3], out_dir_vec_bc[:, :, 34 * 3 : 35 * 3], dim = 2)
                beat_vec = torch.cat((out_dir_vec_bc, left_palm, right_palm), dim = 2)
                beat_vec = F.normalize(beat_vec, dim = -1)
                all_vec = beat_vec.reshape(beat_vec.shape[0] * beat_vec.shape[1], -1, 3)
                
                for idx, pair in enumerate(angle_pair):
                    vec1 = all_vec[:, pair[0]]
                    vec2 = all_vec[:, pair[1]]
                    inner_product = torch.einsum('ij,ij->i', [vec1, vec2])
                    inner_product = torch.clamp(inner_product, -1, 1, out=None)
                    angle = torch.acos(inner_product) / math.pi
                    angle_time = angle.reshape(batch_size, -1)
                    if idx == 0:
                        angle_diff = torch.abs(angle_time[:, 1:] - angle_time[:, :-1]) / change_angle[idx] / len(change_angle)
                    else:
                        angle_diff += torch.abs(angle_time[:, 1:] - angle_time[:, :-1]) / change_angle[idx] / len(change_angle)
                angle_diff = torch.cat((torch.zeros(batch_size, 1).to(device), angle_diff), dim = -1)
                
                for b in range(batch_size):
                    motion_beat_time = []
                    for t in range(2, 33):
                        if (angle_diff[b][t] < angle_diff[b][t - 1] and angle_diff[b][t] < angle_diff[b][t + 1]):
                            if (angle_diff[b][t - 1] - angle_diff[b][t] >= thres or angle_diff[b][t + 1] - angle_diff[b][t] >= thres):
                                motion_beat_time.append(float(t) / 15.0)
                    if (len(motion_beat_time) == 0):
                        continue
                    audio = in_audio[b].cpu().numpy()
                    audio_beat_time = librosa.onset.onset_detect(y=audio, sr=16000, units='time')
                    sum = 0
                    for audio in audio_beat_time:
                        sum += np.power(math.e, -np.min(np.power((audio - motion_beat_time), 2)) / (2 * sigma * sigma))
                    bc.update(sum / len(audio_beat_time), len(audio_beat_time))

                if args.model != 'gesture_autoencoder':
                    if embed_space_evaluator:
                        if not (condition_mode or direct_rl):
                            embed_space_evaluator.push_samples(in_text_padded, in_audio, out_dir_vec, target)
                        else:
                            print(real_out_vec.shape)
                            real_out_vec_eval = real_out_vec.clone().cpu().numpy()
                            real_dir_vec_eval = real_dir_vec.clone().cpu().numpy()
                            unit_real_out = torch.from_numpy(convert_dir_vec_to_norm_vec(real_out_vec_eval)).to(device)
                            unit_dir_vec = torch.from_numpy(convert_dir_vec_to_norm_vec(real_dir_vec_eval)).to(device)
                            unit_real_out = unit_real_out.reshape(unit_real_out.shape[0],unit_real_out.shape[1],-1)
                            unit_dir_vec = unit_dir_vec.reshape(unit_real_out.shape[0],unit_real_out.shape[1],-1)
                            print(unit_dir_vec.shape)
                
                            embed_space_evaluator.push_samples(in_text_padded, in_audio, unit_real_out,unit_dir_vec)

                    # calculate MAE of joint coordinates
                    if not (condition_mode or direct_rl):
                        out_dir_vec = out_dir_vec.cpu().numpy()
                        out_dir_vec += np.array(args.mean_dir_vec).squeeze()
                        out_joint_poses = convert_dir_vec_to_pose(out_dir_vec)
                        target_vec = target_vec.cpu().numpy()
                        target_vec += np.array(args.mean_dir_vec).squeeze()
                        target_poses = convert_dir_vec_to_pose(target_vec)

                        if out_joint_poses.shape[1] == args.n_poses:
                            diff = out_joint_poses[:, args.n_pre_poses:] - target_poses[:, args.n_pre_poses:]
                        else:
                            diff = out_joint_poses - target_poses[:, args.n_pre_poses:]
                        mae_val = np.mean(np.absolute(diff))
                        joint_mae.update(mae_val, batch_size)

                        # accel
                        target_acc = np.diff(target_poses, n=2, axis=1)
                        out_acc = np.diff(out_joint_poses, n=2, axis=1)
                        accel.update(np.mean(np.abs(target_acc - out_acc)), batch_size)
                    else:
                        real_out_dir_vec = real_out_vec.cpu().numpy()
                        out_joint_poses = convert_dir_vec_to_pose_fixed(real_out_dir_vec)
                        
                        target_vec = real_dir_vec.cpu().numpy()
                        target_poses = convert_dir_vec_to_pose_fixed(target_vec)

                        if out_joint_poses.shape[1] == args.n_poses:
                            diff = out_joint_poses[:, args.n_pre_poses:] - target_poses[:, args.n_pre_poses:]
                        else:
                            diff = out_joint_poses - target_poses[:, args.n_pre_poses:]
                        mae_val = np.mean(np.absolute(diff))
                        joint_mae.update(mae_val, batch_size)

                        # accel
                        target_acc = np.diff(target_poses, n=2, axis=1)
                        out_acc = np.diff(out_joint_poses, n=2, axis=1)
                        accel.update(np.mean(np.abs(target_acc - out_acc)), batch_size)

    # print
    # ret_dict = {'joint_mae': joint_mae.avg}
    # elapsed_time = time.time() - start
    # if embed_space_evaluator and embed_space_evaluator.get_no_of_samples() > 0:
    #     frechet_dist, feat_dist = embed_space_evaluator.get_scores()
    #     diversity_score = embed_space_evaluator.get_diversity_scores()
    #     logging.info(
    #         '[VAL] joint mae: {:.5f}, accel diff: {:.5f}, FGD: {:.3f}, diversity_score: {:.3f}, BC: {:.3f}, feat_D: {:.3f} / {:.1f}s'.format(
    #             joint_mae.avg, accel.avg, frechet_dist, diversity_score, bc.avg, feat_dist, elapsed_time))
    #     ret_dict['frechet'] = frechet_dist
    #     ret_dict['feat_dist'] = feat_dist
    #     ret_dict['diversity_score'] = diversity_score
    #     ret_dict['bc'] = bc.avg

    # return ret_dict
    if only_FGD:
        ret_dict = {}
        if embed_space_evaluator and embed_space_evaluator.get_no_of_samples() > 0:
            frechet_dist, feat_dist = embed_space_evaluator.get_scores()
            print(frechet_dist)
            diversity_score = embed_space_evaluator.get_diversity_scores()
            logging.info(
                '[VAL] FGD: {:.3f}, diversity_score: {:.3f}'.format(
                    frechet_dist, diversity_score))
            ret_dict['frechet'] = frechet_dist
        return ret_dict
    else:
        ret_dict = {'joint_mae': joint_mae.avg}
        elapsed_time = time.time() - start
        if embed_space_evaluator and embed_space_evaluator.get_no_of_samples() > 0:
            frechet_dist, feat_dist = embed_space_evaluator.get_scores()
            diversity_score = embed_space_evaluator.get_diversity_scores()
            logging.info(
                '[VAL] joint mae: {:.5f}, accel diff: {:.5f}, FGD: {:.3f}, diversity_score: {:.3f}, BC: {:.3f}, feat_D: {:.3f} / {:.1f}s'.format(
                    joint_mae.avg, accel.avg, frechet_dist, diversity_score, bc.avg, feat_dist, elapsed_time))
            ret_dict['frechet'] = frechet_dist
            ret_dict['feat_dist'] = feat_dist
            ret_dict['diversity_score'] = diversity_score
            ret_dict['bc'] = bc.avg

        return ret_dict

def evaluate_testset_vqvae(test_data_loader, vqvae, embed_space_evaluator, args, pose_dim):

    if embed_space_evaluator:
        embed_space_evaluator.reset()
    # losses = AverageMeter('loss')
    joint_mae = AverageMeter('mae_on_joint')
    accel = AverageMeter('accel')
    bc = AverageMeter('bc')
    start = time.time()

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            print("testing {}/{}".format(iter_idx, len(test_data_loader)))
            in_text, _, in_text_padded, _, target_vec, in_audio, in_spec, _ = data
            batch_size = target_vec.size(0)
            
        
            in_text = in_text.to(device)
            in_text_padded = in_text_padded.to(device)
            in_audio = in_audio.to(device)
            in_spec = in_spec.to(device)
            target = target_vec.to(device)

            pre_seq = target.new_zeros((target.shape[0], target.shape[1], target.shape[2] + 1))
            pre_seq[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses]
            pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

            out_dir_vec,_ = vqvae(target)

            out_dir_vec_bc = out_dir_vec + torch.tensor(args.mean_dir_vec).squeeze(1).unsqueeze(0).unsqueeze(0).to(device)
            left_palm = torch.cross(out_dir_vec_bc[:, :, 11 * 3 : 12 * 3], out_dir_vec_bc[:, :, 17 * 3 : 18 * 3], dim = 2)
            right_palm = torch.cross(out_dir_vec_bc[:, :, 28 * 3 : 29 * 3], out_dir_vec_bc[:, :, 34 * 3 : 35 * 3], dim = 2)
            beat_vec = torch.cat((out_dir_vec_bc, left_palm, right_palm), dim = 2)
            beat_vec = F.normalize(beat_vec, dim = -1)
            all_vec = beat_vec.reshape(beat_vec.shape[0] * beat_vec.shape[1], -1, 3)
            
            for idx, pair in enumerate(angle_pair):
                vec1 = all_vec[:, pair[0]]
                vec2 = all_vec[:, pair[1]]
                inner_product = torch.einsum('ij,ij->i', [vec1, vec2])
                inner_product = torch.clamp(inner_product, -1, 1, out=None)
                angle = torch.acos(inner_product) / math.pi
                angle_time = angle.reshape(batch_size, -1)
                if idx == 0:
                    angle_diff = torch.abs(angle_time[:, 1:] - angle_time[:, :-1]) / change_angle[idx] / len(change_angle)
                else:
                    angle_diff += torch.abs(angle_time[:, 1:] - angle_time[:, :-1]) / change_angle[idx] / len(change_angle)
            angle_diff = torch.cat((torch.zeros(batch_size, 1).to(device), angle_diff), dim = -1)
            
            for b in range(batch_size):
                motion_beat_time = []
                for t in range(2, 33):
                    if (angle_diff[b][t] < angle_diff[b][t - 1] and angle_diff[b][t] < angle_diff[b][t + 1]):
                        if (angle_diff[b][t - 1] - angle_diff[b][t] >= thres or angle_diff[b][t + 1] - angle_diff[b][t] >= thres):
                            motion_beat_time.append(float(t) / 15.0)
                if (len(motion_beat_time) == 0):
                    continue
                audio = in_audio[b].cpu().numpy()
                audio_beat_time = librosa.onset.onset_detect(y=audio, sr=16000, units='time')
                sum = 0
                for audio in audio_beat_time:
                    sum += np.power(math.e, -np.min(np.power((audio - motion_beat_time), 2)) / (2 * sigma * sigma))
                bc.update(sum / len(audio_beat_time), len(audio_beat_time))

            if args.model != 'gesture_autoencoder':
                if embed_space_evaluator:
                    embed_space_evaluator.push_samples(in_text_padded, in_audio, out_dir_vec, target)

                # calculate MAE of joint coordinates
                out_dir_vec = out_dir_vec.cpu().numpy()
                out_dir_vec += np.array(args.mean_dir_vec).squeeze()
                out_joint_poses = convert_dir_vec_to_pose(out_dir_vec)
                target_vec = target_vec.cpu().numpy()
                target_vec += np.array(args.mean_dir_vec).squeeze()
                target_poses = convert_dir_vec_to_pose(target_vec)

                if out_joint_poses.shape[1] == args.n_poses:
                    diff = out_joint_poses[:, args.n_pre_poses:] - target_poses[:, args.n_pre_poses:]
                else:
                    diff = out_joint_poses - target_poses[:, args.n_pre_poses:]
                mae_val = np.mean(np.absolute(diff))
                joint_mae.update(mae_val, batch_size)

                # accel
                target_acc = np.diff(target_poses, n=2, axis=1)
                out_acc = np.diff(out_joint_poses, n=2, axis=1)
                accel.update(np.mean(np.abs(target_acc - out_acc)), batch_size)

    # print
    ret_dict = {'joint_mae': joint_mae.avg}
    elapsed_time = time.time() - start
    if embed_space_evaluator and embed_space_evaluator.get_no_of_samples() > 0:
        frechet_dist, feat_dist = embed_space_evaluator.get_scores()
        diversity_score = embed_space_evaluator.get_diversity_scores()
        logging.info(
            '[VAL] joint mae: {:.5f}, accel diff: {:.5f}, FGD: {:.3f}, diversity_score: {:.3f}, BC: {:.3f}, feat_D: {:.3f} / {:.1f}s'.format(
                joint_mae.avg, accel.avg, frechet_dist, diversity_score, bc.avg, feat_dist, elapsed_time))
        ret_dict['frechet'] = frechet_dist
        ret_dict['feat_dist'] = feat_dist
        ret_dict['diversity_score'] = diversity_score
        ret_dict['bc'] = bc.avg

    return ret_dict

def evaluate_testset_save_video(test_data_loader, diffusion, args, lang_model, pose_dim, repainting=True):

    n_save = 15

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            print("testing {}/{}".format(iter_idx, len(test_data_loader)))
            if condition_mode or direct_rl:
                _, _, in_text_padded, poses, target_vec, in_audio, in_spec, aux_info,real_dir_vec = data
            else:
                _, _, in_text_padded, poses, target_vec, in_audio, in_spec, aux_info = data
    
            # prepare
            select_index = 0

            in_text_padded = in_text_padded[select_index, :].unsqueeze(0).to(device)
            in_audio = in_audio[select_index, :].unsqueeze(0).to(device)
            in_spec = in_spec[select_index, :].unsqueeze(0).to(device)
            target_dir_vec = target_vec[select_index, :].unsqueeze(0).to(device)
            
            if condition_mode or direct_rl:
                real_dir_vec = real_dir_vec[select_index, :].unsqueeze(0).to(device)
            
            in_audio = in_audio.detach().cpu().numpy()
            
            if condition_mode:
                vqcond = CondVQModel(in_channels=126,n_embed=1024,embed_dim=128,hidden_channels=128,num_res_blocks=2).to(device)
                # weight = torch.load('/home/tiger/nfs/workspace/DiffGesture/output/paper_vqvae_cond_300/pose_diffusion_checkpoint_499.bin')["state_dict"]
                # weight = torch.load('/home/tiger/nfs/workspace/DiffGesture/output/fixed_paper_train_vqvae_expressive_cond_best/pose_diffusion_checkpoint_499.bin')["state_dict"]
                weight = torch.load('/data/longbinji/checkpoints_c2g2_best/expressive/paper_vqvae_cond_300/pose_diffusion_checkpoint_499.bin')["state_dict"]
                vqcond.load_state_dict(weight,strict=True)
                
            if combine_mel:
                audio_features = []
                for audio in in_audio:
                    FPS = 60
                    HOP_LENGTH = 512
                    SR = FPS * HOP_LENGTH
                    mfcc = librosa.feature.mfcc(audio, sr=SR, n_mfcc=20).T
                    envelope = librosa.onset.onset_strength(audio, sr=SR)
                    peak_idxs = librosa.onset.onset_detect(
                        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH)
                    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
                    peak_onehot[peak_idxs] = 1.0  # (seq_len,)

                    tempo, beat_idxs = librosa.beat.beat_track(
                        onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH,
                        tightness=100)
                    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
                    beat_onehot[beat_idxs] = 1.0  # (seq_len,)
                    audio_feature = np.concatenate([
                        envelope[:, None], mfcc,peak_onehot[:, None], beat_onehot[:, None]
                            ], axis=-1)
                    audio_features.append(audio_feature)
                audio_features = np.stack(audio_features,0)

            pre_seq = target_dir_vec.new_zeros((target_dir_vec.shape[0], target_dir_vec.shape[1], target_dir_vec.shape[2] + 1))
            pre_seq[:, 0:args.n_pre_poses, :-1] = target_dir_vec[:, 0:args.n_pre_poses]
            pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
            
            # target_nnm = target_nnm[select_index,:].unsqueeze(0).to(device)

            if direct_rl:
                in_audio = torch.from_numpy(in_audio).to(device)
                pre_seq_rl = real_dir_vec.new_zeros((real_dir_vec.shape[0], real_dir_vec.shape[1], real_dir_vec.shape[2] + 1))
                pre_seq_rl[:, 0:args.n_pre_poses, :-1] = real_dir_vec[:, 0:args.n_pre_poses]
                pre_seq_rl[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
                real_out_vec = diffusion.sample(pose_dim, pre_seq_rl, in_audio)
                real_out_seq = real_out_vec[0, :, :].data.cpu().numpy()
                real_dir_vec = np.squeeze(real_dir_vec.cpu().numpy())

            else:
                if args.model != 'pose_diffusion':
                    out_dir_vec = diffusion.sample(pose_dim, pre_seq, in_audio)
                    # print("ddim_sampling !!!")
                    # out_dir_vec,_ = diffusion.ddim_sample(pose_dim, pre_seq, in_audio)
                else:
                    if mode_model == "vqvae":
                        in_audio = torch.from_numpy(in_audio).to(device)
                        if condition_mode:
                            # real_out_vec,_ = diffusion(target,real_dir_vec[:,0,:])
                            # out_dir_vec,latent = diffusion(target)
                            h = diffusion.encoder(target_dir_vec)
                            h = h.transpose(1,2)
                            latent = diffusion.quant_conv(h).transpose(1,2)
                            print("start to predict real")
                            with torch.no_grad():
                                quant,_,_ = vqcond.quantize(latent)
                                real_out_vec = vqcond.decode(quant,real_dir_vec[:,0,:])
                                real_out_seq = real_out_vec[0, :, :].data.cpu().numpy()
                                real_dir_vec = np.squeeze(real_dir_vec.cpu().numpy())

                        else:
                            out_dir_vec,_ = diffusion(target_dir_vec)
                    else:
                        
                        if mel_feature:
                            in_audio = torch.from_numpy(in_audio).to(device)
                            audio_features = torch.from_numpy(audio_features).to(device)
                            print("ddim_sampling !!!")
                            out_dir_vec,_ = diffusion.sample(pose_dim, pre_seq, audio_features)
                            # out_dir_vec,_ = diffusion.sample(pose_dim, pre_seq, audio_features)
                        else:
                            if combine_mel:
                                in_audio = torch.from_numpy(in_audio).to(device)
                                audio_features = torch.from_numpy(audio_features).to(device)
                                out_dir_vec,latent = diffusion.sample(pose_dim, pre_seq,in_audio, audio_feature = audio_features)
                            else:
                                if mode_model == "cond_vqvae":
                                    out_dir_vec,_ = diffusion(target_dir_vec,target_nnm[:,0,:])
                                elif mode_model == "pose_diffusion":
                                    in_audio = torch.from_numpy(in_audio).to(device)
                                    out_dir_vec = diffusion.sample(pose_dim, pre_seq, in_audio)
                                    
                                else:
                                    if repainting:
                                        chunk_target_vec = target_dir_vec[select_index,:,:]
                                        mask = torch.zeros((target_dir_vec.shape[1],128))
                                        mask_start = 6
                                        mask[-mask_start:,:] = 1.0
                                        mask = mask.to(device).float()
                                        chunk_target_vec = chunk_target_vec.to(device)
                                        in_audio = torch.from_numpy(in_audio).to(device)
                                        out_dir_vec,latent = diffusion.sample(pose_dim, pre_seq, in_audio, target_vec=chunk_target_vec,mask = mask,use_repaint=True)
                                        # out_dir_vec[:,-2:,:] = target_dir_vec[:,-2:,:]
                                    else:
                                        in_audio = torch.from_numpy(in_audio).to(device)
                                        out_dir_vec,latent = diffusion.sample(pose_dim, pre_seq, in_audio)
                                        # print("ddim_sampling !!!")
                                    # out_dir_vec,_ = diffusion.ddim_sample(pose_dim, pre_seq, in_audio)
                        if condition_mode:
                            print("start to predict real")
                            with torch.no_grad():
                                quant,_,_ = vqcond.quantize(latent)
                                real_out_vec = vqcond.decode(quant,real_dir_vec[:,0,:])
                                real_out_seq = real_out_vec[0, :, :].data.cpu().numpy()
                                real_dir_vec = np.squeeze(real_dir_vec.cpu().numpy())
                     
            


            # to video
            if iter_idx >= n_save:  # save N samples
                break
        

            audio_npy = np.squeeze(in_audio.cpu().numpy())

            if not (direct_rl or condition_mode):
                target_dir_vec = np.squeeze(target_dir_vec.cpu().numpy())
                out_dir_vec = np.squeeze(out_dir_vec.cpu().numpy())
            
            if mode_model == "cond_vqvae":
                target_nnm = np.squeeze(target_nnm.cpu().numpy())

            input_words = []
            for i in range(in_text_padded.shape[1]):
                word_idx = int(in_text_padded.data[0, i])
                if word_idx > 0:
                    input_words.append(lang_model.index2word[word_idx])
            sentence = ' '.join(input_words)

            aux_str = '({}, time: {}-{})'.format(
                aux_info['vid'][0],
                str(datetime.timedelta(seconds=aux_info['start_time'][0].item())),
                str(datetime.timedelta(seconds=aux_info['end_time'][0].item())))

            mean_data = np.array(args.mean_dir_vec).reshape(-1, 3)
            save_path = args.model_save_path
            
            if mode_model == "cond_vqvae":
                create_video_and_save(
                    save_path, iter_idx, 'short',
                    target_nnm, out_dir_vec, mean_data,
                    sentence, audio=audio_npy, aux_str=aux_str)
            else:
                if condition_mode or direct_rl:
                    create_video_and_save(
                        save_path, iter_idx, 'short',
                        real_dir_vec, real_out_seq, mean_data,
                        sentence, audio=audio_npy, aux_str=aux_str)
                else:
                    create_video_and_save(
                        save_path, iter_idx, 'short',
                        target_dir_vec, out_dir_vec, mean_data,
                        sentence, audio=audio_npy, aux_str=aux_str)

import matplotlib.pyplot as plt
from textwrap import wrap
import matplotlib.animation as animation
import subprocess

def create_video_and_save(save_path, iter_idx, prefix, target, output, mean_data, title,
                          audio=None, aux_str=None, clipping_to_shortest_stream=False, delete_audio_file=True,single=False):
    print('rendering a video...')
    start = time.time()
    if single:
        fig = plt.figure(figsize=(6, 4))
        axes = [fig.add_subplot(1, 1, 1, projection='3d')]
        axes[0].view_init(elev=20, azim=-60)
    else:
        fig = plt.figure(figsize=(8, 4))
        axes = [fig.add_subplot(1, 2, 1, projection='3d'), fig.add_subplot(1, 2, 2, projection='3d')]
        axes[0].view_init(elev=20, azim=-60)
        axes[1].view_init(elev=20, azim=-60)

    
    fig_title = title

    if aux_str:
        fig_title += ('\n' + aux_str)
    fig.suptitle('\n'.join(wrap(fig_title, 75)), fontsize='medium')
    
    if not (condition_mode or direct_rl):
        print("add mean")
        # un-normalization and convert to poses
        mean_data = mean_data.flatten()
        output = output + mean_data
        output_poses = convert_dir_vec_to_pose(output)
        target_poses = None
        if target is not None:
            target = target + mean_data
            target_poses = convert_dir_vec_to_pose(target)
    else:
        output_poses = convert_dir_vec_to_pose_fixed(output)
        target_poses = convert_dir_vec_to_pose_fixed(target)
        
    def animate_single(i):
        for k, name in enumerate(['human']):
                if name == 'human' and target is not None and i < len(target):
                    pose = target_poses[i]
                elif name == 'generated' and i < len(output):
                    pose = output_poses[i]
                else:
                    pose = None

                if pose is not None:
                    axes[k].clear()
                    for j, pair in enumerate(dir_vec_pairs):
                        axes[k].plot([pose[pair[0], 0], pose[pair[1], 0]],
                                    [pose[pair[0], 2], pose[pair[1], 2]],
                                    [pose[pair[0], 1], pose[pair[1], 1]],
                                    zdir='z', linewidth=1.5)
                    axes[k].set_xlim3d(-0.5, 0.5)
                    axes[k].set_ylim3d(0.5, -0.5)
                    axes[k].set_zlim3d(0.5, -0.5)
                    axes[k].set_xlabel('x')
                    axes[k].set_ylabel('z')
                    axes[k].set_zlabel('y')
                    axes[k].set_title('{} ({}/{})'.format(name, i + 1, len(output)))
                    # axes[k].axis('off')
    

    def animate(i):
        for k, name in enumerate(['human', 'generated']):
            if name == 'human' and target is not None and i < len(target):
                pose = target_poses[i]
            elif name == 'generated' and i < len(output):
                pose = output_poses[i]
            else:
                pose = None

            if pose is not None:
                axes[k].clear()
                for j, pair in enumerate(dir_vec_pairs):
                    axes[k].plot([pose[pair[0], 0], pose[pair[1], 0]],
                                 [pose[pair[0], 2], pose[pair[1], 2]],
                                 [pose[pair[0], 1], pose[pair[1], 1]],
                                 zdir='z', linewidth=1.5)
                axes[k].set_xlim3d(-0.5, 0.5)
                axes[k].set_ylim3d(0.5, -0.5)
                axes[k].set_zlim3d(0.5, -0.5)
                axes[k].set_xlabel('x')
                axes[k].set_ylabel('z')
                axes[k].set_zlabel('y')
                axes[k].set_title('{} ({}/{})'.format(name, i + 1, len(output)))
                axes[k].axis('off')

    if target is not None:
        num_frames = max(len(target), len(output))
    else:
        num_frames = len(output)
    
    if single:
        prefix = "edited_gt"
        ani = animation.FuncAnimation(fig, animate_single, interval=30, frames=num_frames, repeat=False)
    else:
        ani = animation.FuncAnimation(fig, animate, interval=30, frames=num_frames, repeat=False)

    # show audio
    audio_path = None
    if audio is not None:
        assert len(audio.shape) == 1  # 1-channel, raw signal
        audio = audio.astype(np.float32)
        sr = 16000
        audio_path = '{}/{}.wav'.format(save_path, iter_idx)
        sf.write(audio_path, audio, sr)

    # save video
    try:
        video_path = '{}/temp_{}.mp4'.format(save_path,  iter_idx)
        ani.save(video_path, fps=15, dpi=80)  # dpi 150 for a higher resolution
        del ani
        plt.close(fig)
    except RuntimeError:
        assert False, 'RuntimeError'

    # merge audio and video
    if audio is not None:
        merged_video_path = '{}/{}_{}.mp4'.format(save_path, prefix, iter_idx)
        cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', video_path, '-i', audio_path, '-strict', '-2',
               merged_video_path]
        if clipping_to_shortest_stream:
            cmd.insert(len(cmd) - 1, '-shortest')
        subprocess.call(cmd)
        if delete_audio_file:
            os.remove(audio_path)
        os.remove(video_path)

    print('done, took {:.1f} seconds'.format(time.time() - start))
    return output_poses, target_poses

def main(mode, checkpoint_path):
    
    if mode_model == "vqvae" or mode_model == "cond_vqvae":
        args, vqvae, lang_model, speaker_model, pose_dim = load_checkpoint_and_vqmodel(
            checkpoint_path, device)
    elif mode_model == "latent_diffusion":
        args, latent_diffusion, lang_model, speaker_model, pose_dim = load_checkpoint_and_latent_model(
            checkpoint_path, device)
    else:
        args, diffusion, lang_model, speaker_model, pose_dim = load_checkpoint_and_model(
            checkpoint_path, device)

    # random seed
    if args.random_seed >= 0:
        set_random_seed(args.random_seed)
        
    set_random_seed(10)

    # set logger
    set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs, default {}".format(torch.cuda.device_count(), device))
    logging.info(pprint.pformat(vars(args)))

    # load mean vec
    mean_pose = np.array(args.mean_pose).squeeze()
    mean_dir_vec = np.array(args.mean_dir_vec).squeeze()

    # load lang_model
    vocab_cache_path = os.path.join('/data/longbinji/ted_expressive_pickle', 'vocab_cache.pkl')
    with open(vocab_cache_path, 'rb') as f:
        lang_model = pickle.load(f)

    collate_fn = default_collate_fn

    def load_dataset(path):
        dataset = SpeechMotionDataset(path,
                                      n_poses=args.n_poses,
                                      subdivision_stride=args.subdivision_stride,
                                      pose_resampling_fps=args.motion_resampling_framerate,
                                      speaker_model=speaker_model,
                                      mean_pose=mean_pose,
                                      mean_dir_vec=mean_dir_vec,
                                      use_cond = (condition_mode or direct_rl)
                                      )
        print(len(dataset))
        return dataset

    if mode == 'eval':
        val_data_path = '/data/longbinji/ted_expressive_pickle/val_pickle'
        eval_net_path = 'output/TED_Expressive_output/AE-cos1e-3/checkpoint_best.bin'
        embed_space_evaluator = EmbeddingSpaceEvaluator(args, eval_net_path, lang_model, device)
        val_dataset = load_dataset(val_data_path)
        data_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size*2, collate_fn=collate_fn,
                                 shuffle=False, drop_last=True, num_workers=args.loader_workers)
        val_dataset.set_lang_model(lang_model)
        if mode_model == "vqvae":
            evaluate_testset(data_loader, vqvae, embed_space_evaluator, args, pose_dim)
        else:
            if mode_model == "pose_diffusion":
                evaluate_testset(data_loader, diffusion, embed_space_evaluator, args, pose_dim)
            else:
                evaluate_testset(data_loader, latent_diffusion, embed_space_evaluator, args, pose_dim)
    
    elif mode == 'short':
        val_data_path = '/data/longbinji/ted_expressive_pickle/val_pickle'
        val_dataset = load_dataset(val_data_path)
        data_loader = DataLoader(dataset=val_dataset, batch_size=32, collate_fn=collate_fn,
                                 shuffle=False, drop_last=True, num_workers=args.loader_workers)
        val_dataset.set_lang_model(lang_model)
        if mode_model == "vqvae" or mode_model=="cond_vqvae":
            evaluate_testset_save_video(data_loader, vqvae, args, lang_model, pose_dim)
        elif mode_model == "latent_diffusion":
            evaluate_testset_save_video(data_loader, latent_diffusion, args, lang_model, pose_dim)
        elif mode_model == "pose_diffusion":
            evaluate_testset_save_video(data_loader, diffusion, args, lang_model, pose_dim)

    elif mode == 'long':
        clip_duration_range = [10, 15]

        n_generations = 5

        # load clips and make gestures
        n_saved = 0
        lmdb_env = lmdb.open('/home/tiger/nfs/workspace/ted_expressive/val_pickle', readonly=True, lock=False)

        with lmdb_env.begin(write=False) as txn:
            keys = [key for key, _ in txn.cursor()]
            while n_saved < n_generations:  # loop until we get the desired number of results
                # select video
                key = random.choice(keys)

                buf = txn.get(key)
                video = pickle.loads(buf)
                vid = video['vid']
                clips = video['clips']

                # select clip
                n_clips = len(clips)
                if n_clips == 0:
                    continue
                clip_idx = random.randrange(n_clips)

                clip_poses = clips[clip_idx]['skeletons_3d']
                clip_audio = clips[clip_idx]['audio_raw']
                clip_words = clips[clip_idx]['words']
                clip_time = [clips[clip_idx]['start_time'], clips[clip_idx]['end_time']]

                clip_poses = resample_pose_seq(clip_poses, clip_time[1] - clip_time[0],
                                                                args.motion_resampling_framerate)
                target_dir_vec = convert_pose_seq_to_dir_vec(clip_poses)
                
                real_dir_vec = convert_pose_seq_to_dir_vec(clip_poses,norm=False)
                real_dir_vec = real_dir_vec.reshape(target_dir_vec.shape[0], -1)
                
                target_dir_vec = target_dir_vec.reshape(target_dir_vec.shape[0], -1)
                target_dir_vec -= mean_dir_vec

                # check duration
                clip_duration = clip_time[1] - clip_time[0]
                if clip_duration < clip_duration_range[0] or clip_duration > clip_duration_range[1]:
                    continue

                # synthesize
                for selected_vi in range(len(clip_words)):  # make start time of input text zero
                    clip_words[selected_vi][1] -= clip_time[0]  # start time
                    clip_words[selected_vi][2] -= clip_time[0]  # end time
                    
                if mode_model == "latent_diffusion":
                    out_dir_vec = generate_gestures(args, latent_diffusion, lang_model, clip_audio, clip_words, pose_dim, last_resample=repainting_only,
                                                    seed_seq=target_dir_vec[0:args.n_pre_poses], fade_out=False, target_vec = target_dir_vec, real_dir_vec = real_dir_vec)
                elif mode_model == "pose_diffusion":
                    out_dir_vec = generate_gestures(args, diffusion, lang_model, clip_audio, clip_words, pose_dim, last_resample=repainting_only,
                                                    seed_seq=target_dir_vec[0:args.n_pre_poses], fade_out=False, target_vec = target_dir_vec, real_dir_vec = real_dir_vec)
                    
                print(out_dir_vec.shape,target_dir_vec.shape)

                # make a video
                aux_str = '({}, time: {}-{})'.format(vid, str(datetime.timedelta(seconds=clip_time[0])),
                                                     str(datetime.timedelta(seconds=clip_time[1])))
                mean_data = np.array(args.mean_dir_vec).reshape(-1, 3)
                save_path = args.model_save_path
                if condition_mode:
                    print("save real")
                    create_video_and_save(
                        save_path, n_saved, 'long',
                        real_dir_vec, out_dir_vec, mean_data,
                        '', audio=clip_audio, aux_str=aux_str)
                else:
                    create_video_and_save(
                        save_path, n_saved, 'long',
                        target_dir_vec, out_dir_vec, mean_data,
                        '', audio=clip_audio, aux_str=aux_str)
                n_saved += 1

    else:
        assert False, 'wrong mode'


if __name__ == '__main__':
    mode = sys.argv[1]
    mode_model = sys.argv[2]
    condition_mode = sys.argv[3]
    repainting_only = sys.argv[4]
    
    assert mode_model in ["vqvae","latent_diffusion","pose_diffusion"]
    
    # condition_mode = False
    mel_feature = False
    combine_mel = False
    repainting_only = False
    direct_rl = False
    # mode_model = 'cond_vqvae'
    assert mode in ["eval", "short", "long", "eval_vq"]
    

    ckpt_path = 'train_diffusion_expressive/pose_diffusion_checkpoint_499.bin'
    ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/train_vqvae_expressive_v2/pose_diffusion_checkpoint_300.bin'
    ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/train_latent_diffusion_300_vqvae_v2/pose_diffusion_checkpoint_499.bin' # 300 best
    # # ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/train_expressive_latent_diffusion/pose_diffusion_checkpoint_499.bin'
    # ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/train_expressive_latent_wo_decoder/pose_diffusion_checkpoint_480.bin'
    # # ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/train_expressive_latent_diffusion/pose_diffusion_checkpoint_150.bin'
    
    # # latent diffusion without decoder
    # ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/train_expressive_latent_low_lr/pose_diffusion_checkpoint_300.bin'
    
    # ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/train_latent_diffusion_expressive_low_lr/pose_diffusion_checkpoint_499.bin'
    
    # # speaker conditional vqvae
    # # ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/train_vqvae_cond/pose_diffusion_checkpoint_350.bin'
    ckpt_path  = '/home/tiger/nfs/workspace/DiffGesture/output/paper_train_latent_300_low_lr/pose_diffusion_checkpoint_499.bin'
    
    ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/paper_train_vqvae_expressive_with_val_fixed/pose_diffusion_checkpoint_best.bin'
    ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/paper_train_latent_300_low_lr/pose_diffusion_checkpoint_499.bin'
    ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/paper_train_latent_300_lower_lr/pose_diffusion_checkpoint_350.bin'
    # ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/paper_train_latent_300_low_lr_transwav/pose_diffusion_checkpoint_499.bin'
    
    # ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/train_diffusion_expressive_ori/pose_diffusion_checkpoint_499.bin'
    # ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/paper_train_latent_300_low_lr_preseq/pose_diffusion_checkpoint_499.bin'
    ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/paper_train_latent_300_low_lr_lstm/pose_diffusion_checkpoint_450.bin'
    ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/paper_vqvae_cond_300/pose_diffusion_checkpoint_499.bin'
    ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/paper_train_vqvae_expressive_512length_256/pose_diffusion_checkpoint_best.bin'
    ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/fixed_paper_train_vqvae_expressive_fixed/pose_diffusion_checkpoint_best.bin'
    ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/fixed_paper_train_vqvae_expressive_latent_best/pose_diffusion_checkpoint_300.bin'
    ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/paper_train_vqvae_expressive_512length_64/pose_diffusion_checkpoint_best.bin'
    ckpt_path = '/home/tiger/nfs/workspace/checkpoints_c2g2_best/expressive/paper_train_latent_300_lower_lr/pose_diffusion_checkpoint_best.bin'
    
    # ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/train_diffusion_expressive_ori/pose_diffusion_checkpoint_499.bin'
    
    # best latent-diffusion
    ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/paper_train_latent_300_lower_lr/pose_diffusion_checkpoint_350.bin'
    
    # best vqvae
    ckpt_path = '/home/tiger/nfs/workspace/DiffGesture/output/train_vqvae_expressive_v2/pose_diffusion_checkpoint_300.bin'


    # ai002 server: 
    if mode_model == "latent_diffusion":
        ckpt_path = '/data/longbinji/checkpoints_c2g2_best/expressive/paper_train_latent_300_lower_lr/pose_diffusion_checkpoint_best.bin'
    elif mode_model == "vqvae":
        ckpt_path = '/data/longbinji/checkpoints_c2g2_best/expressive/train_vqvae_expressive_v2/pose_diffusion_checkpoint_best.bin'
    
    # best condvq


    # ckpt_path = '/data/longbinji/DiffGesture/output/train_expressive_ori_rl/pose_diffusion_checkpoint_499.bin'
    # ckpt_path = '/data/longbinji/checkpoints_c2g2_best/expressive/train_vqvae_expressive_v2/pose_diffusion_checkpoint_300.bin'
    direct_rl = False
    
    main(mode, ckpt_path)
