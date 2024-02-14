import os
import subprocess
import numpy as np
import torch
import matplotlib
import shutil

import matplotlib.pyplot as plt
from textwrap import wrap
import matplotlib.animation as animation
import subprocess

import sys
sys.path.append("../workspace/expose")
from expose.models.smplx_net import SMPLXNet
from utils.data_utils_expressive import convert_dir_vec_to_pose, convert_pose_seq_to_dir_vec, resample_pose_seq, dir_vec_pairs, convert_dir_vec_to_pose_fixed, convert_dir_vec_to_pose_l2n, convert_dir_vec_to_norm_vec,convert_dir_vec_to_pose_real
from model.pose_diffusion import PoseDiffusion, PoseLatentDiffusion
import lmdb
import random
import pickle
from model.vqvae import VQModel,VAEModel, CondVQModel
from expose.config import cfg
import math
import time
import cv2
import json
import soundfile as sf

pairs_used = [
    (0, 1, 'r'), # 0, spine-neck
    (1, 2, 'r'), # 1, neck-left shoulder
    (1, 3, 'r'), # 2, neck-right shoulder
    (2, 4, 'r'), # 3, left shoulder-elbow
    (4, 6, 'r'), # 4, left elbow-wrist

    (6, 8, 'r'), # 5 wrist-left index 1
    (8, 9, 'r'), # 6
    (9, 10, 'r'), # 7

    (6, 11, 'r'), # 8 wrist-left middle 1
    (11, 12, 'r'), # 9
    (12, 13, 'r'), # 10

    (6, 14, 'r'), # 11 wrist-left pinky 1
    (14, 15, 'r'), # 12
    (15, 16, 'r'), # 13

    (6, 17, 'r'), # 14 wrist-left ring 1
    (17, 18, 'r'), # 15
    (18, 19, 'r'), # 16

    (6, 20, 'r'), # 17 wrist-left thumb 1
    (20, 21, 'r'), # 18
    (21, 22, 'r'), # 19

    (3, 5, 'r'), # 20, right shoulder-elbow
    (5, 7, 'r'), # 21, right elbow-wrist

    (7, 23, 'r'), # 22 wrist-right index 1
    (23, 24, 'r'), # 23
    (24, 25, 'r'), # 24

    (7, 26, 'r'), # 25 wrist-right middle 1
    (26, 27, 'r'), # 26
    (27, 28, 'r'), # 27

    (7, 29, 'r'), # 28 wrist-right pinky 1
    (29, 30, 'r'), # 29
    (30, 31, 'r'), # 30

    (7, 32, 'r'), # 31 wrist-right ring 1
    (32, 33, 'r'), # 32
    (33, 34, 'r'), # 33

    (7, 35, 'r'), # 34 wrist-right thumb 1
    (35, 36, 'r'), # 35
    (36, 37, 'r'), # 36

    (1, 38, 'r'), # 37, neck-nose
    (38, 39, 'r'), # 38, nose-right eye
    (38, 40, 'r'), # 39, nose-left eye
    (39, 41, 'r'), # 40, right eye-right ear
    (40, 42, 'r'), # 41, left eye-left ear
]


OPENPOSE_OPTION = "--face --hand --number_people_max 1 --display 0 --render_pose 0"
BASE_DIR = "/data/longbinji/DiffGesture"
EXPOSE_BASE_DIR = "/data/longbinji/workspace/expose/"
OPENPOSE_BASE_DIR = "/data/longbinji/workspace/openpose"
OPENPOSE_BIN_PATH = "/data/longbinji/workspace/openpose/build/examples/openpose/openpose.bin"
# PARSING_REPO = "/home/tiger/nfs/workspace/Self-Correction-Human-Parsing"

def load_checkpoint_and_model(checkpoint_path, _device='cpu'):
    print('loading checkpoint {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    args = checkpoint['args']
    epoch = checkpoint['epoch']
    lang_model = checkpoint['lang_model']
    speaker_model = checkpoint['speaker_model']
    pose_dim = checkpoint['pose_dim']
    args.vqvae_weight = "output/train_vqvae_expressive_v2/pose_diffusion_checkpoint_300.bin"
    args.device = _device
    print('epoch {}'.format(epoch))

    print("init diffusion model")
    diffusion = PoseLatentDiffusion(args).to(_device)

    diffusion.load_state_dict(checkpoint['state_dict'])

    return args, diffusion, lang_model, speaker_model, pose_dim

def extract_openpose(datapath):
    os.chdir(OPENPOSE_BASE_DIR)
    folder_path = os.path.join(datapath,"images")
    output_dir = os.path.join(datapath,"keypoints")
        
    command = OPENPOSE_BIN_PATH + " " + OPENPOSE_OPTION + " --image_dir \"" + folder_path + "\"" + " --display 0 --render_pose 0"
    # command += " --write_video " + OUTPUT_VIDEO_PATH + "/" + vid + "_result.avi"  # write result video
    command += " --write_json " + output_dir
    print(command)
    subprocess.call(command, shell=True)
    
def extract_expose(datapath):
    expose_out = os.path.join(datapath,"exposes")
    os.chdir(EXPOSE_BASE_DIR)
    frames_num = os.path.join(datapath,"images")
    openpose_path = datapath
    
    command = "python " + EXPOSE_BASE_DIR + "inference.py --exp-cfg " + EXPOSE_BASE_DIR + "data/conf.yaml --datasets openpose --exp-opts datasets.body.batch_size 256 datasets.body.openpose.data_folder " + openpose_path + " --show False --output-folder " + expose_out + " --save-params True --save-vis False --save-mesh False"
    print(command)
    subprocess.call(command, shell=True)
        
def provide_audio(audio_path=None):
    if audio_path==None:
        lmdb_env = lmdb.open('/data/longbinji/ted_expressive_pickle/val_pickle', readonly=True, lock=False)

        with lmdb_env.begin(write=False) as txn:
            keys = [key for key, _ in txn.cursor()]
            
            # select video
            n_clips = 0
            while n_clips == 0:
                key = random.choice(keys)

                buf = txn.get(key)
                video = pickle.loads(buf)
                vid = video['vid']
                clips = video['clips']
            
                # select clip
                n_clips = len(clips)
                
            clip_idx = random.randrange(n_clips)

            clip_audio = clips[clip_idx]['audio_raw']
            audio = clip_audio
    else:
        audio = audio_path
    return audio
            

def upperbody(args,diffusion,pose_dim,expose_path,control_path=None):
    
    images = os.listdir(expose_path)
    if control_path!=None:
        print("control mode with different seed pose")
        control_images = os.listdir(control_path)
        pre_control_images = sorted(control_images)[:4]
    
    openpose_path = os.path.join(os.path.split(expose_path)[-2],"keypoints")
    openposes = sorted(os.listdir(openpose_path))
    with open(os.path.join(openpose_path,openposes[0]),"rb") as f:
            poses = json.load(f)
            nose_openpose = poses["people"][0]["pose_keypoints_2d"][0],poses["people"][0]["pose_keypoints_2d"][1]
    
    
    print(len(images))
    pre_images = sorted(images)[:4]
    
    scale = None
    translation = None
    mm = None
    px = None
    center = None
    full_joints = None
    start_joint = None
    
    poses = []
    
    for i in range(4):
        npz_path = os.path.split(pre_images[i])[-1]+"_params.npz"
        expose_dict = np.load(os.path.join(expose_path,pre_images[i],npz_path))
        if i==0:
            scale = expose_dict["scale"][0]
            print(scale.shape)
            translation = expose_dict["transl"]
            print(translation.shape)
            mm = expose_dict["focal_length_in_mm"]
            px = expose_dict["focal_length_in_px"]
            center = expose_dict["center"]
            full_joints = expose_dict["joints"]
            print(full_joints.shape)
            start_joint = expose_dict["joints"][9]
            
            nose_joint = expose_dict["proj_joints"][55]
            multiply = (nose_openpose-center)[1]/(nose_joint* px*11/ mm)[1]
            print(multiply)
            
        proj_joints = expose_dict["joints"]
        upper_joints = np.vstack((proj_joints[9], proj_joints[12], proj_joints[16:22], proj_joints[25:40], proj_joints[40:55], proj_joints[55:60]))
        poses.append(upper_joints)

    poses = np.stack(poses)

    if control_path:
        control_poses = []
        for i in range(4):
            control_npz_path = os.path.split(pre_control_images[i])[-1]+"_params.npz"
            control_expose_dict = np.load(os.path.join(control_path,pre_control_images[i],control_npz_path))

            control_proj_joints = control_expose_dict["joints"]
            control_upper_joints = np.vstack((control_proj_joints[9], control_proj_joints[12], control_proj_joints[16:22], control_proj_joints[25:40], control_proj_joints[40:55], control_proj_joints[55:60]))
            control_poses.append(control_upper_joints)

        control_poses = np.stack(control_poses)
        
    audio = provide_audio()

    if control_path:
        generated_pose,norm_dir = generate_gestures(args,diffusion,poses,audio,start_joint,control_seed=control_poses)
    else:
        generated_pose,norm_dir = generate_gestures(args,diffusion,poses,audio,start_joint)
   
    print(generated_pose.shape)
    
    # localize upper body keypoints
    local_joints = []
    
    for i in range(len(generated_pose)):
        gene = generated_pose[i]
        translation = translation[:2]
        upper_local = scale.reshape(-1, 1) * (gene[:, :2] + translation.reshape(-1, 2))
        local_joints.append(upper_local)
        
    local_joints = np.stack(local_joints)
    print(local_joints.shape)
    
    
    
    proj_joints = local_joints * px*11/ mm
    proj_joints*=multiply*0.87
    proj_joints += center
    
    
        
    return proj_joints,audio, norm_dir


# def save_jsons(proj_joints,save_path):
#     "people":[{"person_id":[-1],"pose_keypoints_2d":[52
#     for i in range(len(proj_joints)):
#         dict_openpose = {}
#         dict_openpose["version"] = 1.3
#         dict_openpose["people"] = []
#         new_dict = {}
#         new_dict["person_id"] = [-1]
#         new_dict["pose_keypoints_2d"] = np.zeros((75)).tolist()
#         new_dict["face_keypoints_2d"] = np.zeros((210)).tolist()
        

def draw_3d_on_image(img, proj_joints, height, width, thickness=15):
    # if proj_joints == []:
    #     return img

    new_img = img.copy()
    for pair in pairs_used:
        pt1 = (int(proj_joints[pair[0]][0]), int(proj_joints[pair[0]][1]))
        pt2 = (int(proj_joints[pair[1]][0]), int(proj_joints[pair[1]][1]))
        if pt1[0] >= width or pt1[0] <= 0 or pt1[1] >= height or pt1[1] <= 0 or pt2[0] >= width or pt2[0] <= 0 or pt2[1] >= height or pt2[1] <= 0:
            pass
        else:
            rgb = [v * 255 for v in matplotlib.colors.to_rgba(pair[2])][:3]
            cv2.line(new_img, pt1, pt2, color=rgb[::-1], thickness=thickness)

    return new_img

def draw(frame_path,generated,output_folder):
    frames = os.listdir(frame_path)
    frames = sorted(frames)
    os.makedirs(output_folder,exist_ok=True)
    first_image = cv2.imread(os.path.join(frame_path,frames[0]))
    
    
    for i in range(len(generated)):
        if i<=3:
            fake_image = cv2.imread(os.path.join(frame_path,frames[i]))
        else:
            fake_image = np.zeros(first_image.shape)
            fake_image[fake_image==0.0] = 255
            
        fake_image = draw_3d_on_image(fake_image,generated[i],1000,1000,5)
        output_path = os.path.join(output_folder,f"{str(i).zfill(5)}.png")
        cv2.imwrite(output_path,fake_image)
        
def combine_video(audio,output_folder):
    frames = sorted(os.listdir(output_folder))
    first_frame = cv2.imread(os.path.join(output_folder,frames[0]))
    
    video_path = os.path.join(output_folder,"temp_video.mp4")
    fps = 15
    size = first_frame.shape[0],first_frame.shape[1]
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter(video_path, fourcc, fps, size)

    
    for frame in frames:
            image = cv2.imread(os.path.join(output_folder,frame))
            videoWriter.write(image)
    videoWriter.release()
            
    audio_path = os.path.join(output_folder,"temp_raw.wav")
    
    if audio is not None:
        print(audio.shape)
        assert len(audio.shape) == 1  # 1-channel, raw signal
        audio = audio.astype(np.float32)
        sr = 16000
        sf.write(audio_path, audio, sr)
    

    # merge audio and video
    if audio is not None:
        print("start to merge")
        merged_video_path = os.path.join(output_folder,"video.mp4")
        cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', video_path, '-i', audio_path, '-strict', '-2',
               merged_video_path]
        # if clipping_to_shortest_stream:
        #     cmd.insert(len(cmd) - 1, '-shortest')
        subprocess.call(cmd)
        
        # os.remove(audio_path)
        # os.remove(video_path)
        
def create_video_and_save(save_path, iter_idx, prefix, output, title = "test",
                          audio=None, aux_str=None, clipping_to_shortest_stream=False, delete_audio_file=True,single=False):
    os.makedirs(save_path,exist_ok=True)
    target = None
    print('rendering a video...')
    start = time.time()

    fig = plt.figure(figsize=(6, 4))
    axes = [fig.add_subplot(1, 1, 1, projection='3d')]
    axes[0].view_init(elev=20, azim=-60)
    

    
    fig_title = title

    if aux_str:
        fig_title += ('\n' + aux_str)
    fig.suptitle('\n'.join(wrap(fig_title, 75)), fontsize='medium')
    
    
    output_poses = convert_dir_vec_to_pose_l2n(output)
        
    def animate_single(i):
        for k, name in enumerate(['human']):
                pose = output_poses[i]

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
    
    prefix = "edited_gt"
    ani = animation.FuncAnimation(fig, animate_single, interval=30, frames=num_frames, repeat=False)
    

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
    
    
    
    
    

def generate_gestures(args, diffusion, real_pose, audio, start_joint,pose_dim=216, audio_sr=16000,
                      seed_seq=None, fade_out=False, target_vec=None, last_resample=False, pred_real=True,condition_mode=True,control_seed=[]):
    out_list = []
    
    mean_dir_vec = np.array(args.mean_dir_vec).squeeze()
    
    unit_vector = convert_pose_seq_to_dir_vec(real_pose)
    if len(control_seed)>0:
        unit_control = convert_pose_seq_to_dir_vec(control_seed)
    
    real_dir_vec =convert_pose_seq_to_dir_vec(real_pose, norm=False)
   
    unit_vector = unit_vector.reshape(unit_vector.shape[0],-1)-mean_dir_vec
    if len(control_seed)>0:
        unit_control = unit_control.reshape(unit_vector.shape[0],-1)-mean_dir_vec
    real_dir_vec = real_dir_vec.reshape(real_dir_vec.shape[0],-1)

    if len(control_seed)==0:
        seed_seq = unit_vector
        print(unit_vector.shape)
    else:
        seed_seq = unit_control
    
    if condition_mode:
        real_out_list = []
        vqcond = CondVQModel(in_channels=126,n_embed=1024,embed_dim=128,hidden_channels=128,num_res_blocks=2).to(device)
        weight = torch.load('output/paper_vqvae_cond_300/pose_diffusion_checkpoint_499.bin')["state_dict"]
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
                mask_start = int(0.4*unit_length)
                mask[-mask_start:,:] = 1.0
                mask = mask.to(device).float()
                chunk_target_vec = torch.from_numpy(chunk_target_vec).to(device).float()
                if args.model == 'pose_diffusion':
                    out_dir_vec,latent = diffusion.sample(pose_dim, pre_seq, in_audio,chunk_target_vec,mask,use_repaint=True)
                    
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
                out_dir_vec,latent = diffusion.sample(pose_dim, pre_seq, in_audio)

            out_seq = out_dir_vec[0, :, :].data.cpu().numpy()

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
        
    out_pose = convert_dir_vec_to_pose_real(out_dir_vec,start_joint)
    return out_pose, out_dir_vec
    
    
def ffmpeg_video(norm_video,output_folder):
    os.makedirs(output_folder)
    command_ffmpeg = "ffmpeg -i " + norm_video + " -start_number 0 -f image2 " + output_folder +"/" + "frames" + "_%012d" + ".png"
    print(command_ffmpeg)
    subprocess.call(command_ffmpeg, shell=True)
    
    
def main(datapath):
    print(f'start to process for clip {datapath}')
    ckpt_path = 'output/paper_train_latent_300_lower_lr/pose_diffusion_checkpoint_350.bin'
    device = "cuda:0"
    if os.path.exists(os.path.join(datapath,"output")):
        shutil.rmtree(os.path.join(datapath,"output"))
    args, diffusion, lang_model, speaker_model, pose_dim = load_checkpoint_and_model(ckpt_path,device)
    # extract_openpose(datapath)
    # extract_expose(datapath)
    proj_joint,audio,norm_dir = upperbody(args,diffusion,pose_dim,os.path.join(datapath,"exposes"))
    draw(os.path.join(datapath,"images"),proj_joint,os.path.join(datapath,"output"))
    # create_video_and_save(os.path.join(datapath,"output_norm"),1,"demo",norm_dir,audio = audio)
    combine_video(audio,os.path.join(datapath,"output"))

def control_main(datapath,controlpath):
    print(f'start to process for clip {datapath}')
    ckpt_path = 'output/paper_train_latent_300_lower_lr/pose_diffusion_checkpoint_350.bin'
    device = "cuda:0"
    if os.path.exists(os.path.join(datapath,"output")):
        shutil.rmtree(os.path.join(datapath,"output"))
    args, diffusion, lang_model, speaker_model, pose_dim = load_checkpoint_and_model(ckpt_path,device)
    # extract_openpose(datapath)
    # extract_expose(datapath)

    print(f'start to process for control clip {controlpath}')
    
    if os.path.exists(os.path.join(controlpath,"output")):
        shutil.rmtree(os.path.join(controlpath,"output"))
    
    # extract_openpose(controlpath)
    # extract_expose(controlpath)

    os.chdir(BASE_DIR)
    proj_joint,audio,norm_dir = upperbody(args,diffusion,pose_dim,os.path.join(datapath,"exposes"),os.path.join(controlpath,"exposes"))
    print(os.path.join(datapath,"controlled_output"))
    draw(os.path.join(datapath,"images"),proj_joint,os.path.join(datapath,"controlled_output"))
    # create_video_and_save(os.path.join(datapath,"output_norm"),1,"demo",norm_dir,audio = audio)
    combine_video(audio,os.path.join(datapath,"controlled_output"))

if __name__ == '__main__':
    seed_path = sys.argv[1]
    identity_path = sys.argv[2]
    
    if seed_path == identity_path:
        main(seed_path)
    else:
        control_main(identity_path,seed_path)

    
device = "cuda:0"

# main('/data/longbinji/DiffGesture/demos/data/clip3')
# main('/home/tiger/nfs/workspace/all_experiments/demos/data/clip6')
# main('/home/tiger/nfs/workspace/all_experiments/demos/data/clip7')
# main('/home/tiger/nfs/workspace/all_experiments/demos/data/clip8')
control_main('/data/longbinji/DiffGesture/demos/data/clip2','/data/longbinji/DiffGesture/demos/data/clip3')

# ffmpeg_video("/home/tiger/nfs/workspace/DiffGesture/demos/data/clip2/output_norm/edited_gt_1.mp4","/home/tiger/nfs/workspace/DiffGesture/demos/data/clip2/output_norm_frames")
# ffmpeg_video("/home/tiger/nfs/workspace/DiffGesture/demos/data/clip3/output_norm/edited_gt_1.mp4","/home/tiger/nfs/workspace/DiffGesture/demos/data/clip3/output_norm_frames")
# ffmpeg_video("/home/tiger/nfs/workspace/DiffGesture/demos/data/clip4/output_norm/edited_gt_1.mp4","/home/tiger/nfs/workspace/DiffGesture/demos/data/clip4/output_norm_frames")
    