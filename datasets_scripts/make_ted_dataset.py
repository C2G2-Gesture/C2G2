# ------------------------------------------------------------------------------
# Copyright (c) ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------
import glob
import os
import pickle
import sys

import cv2
import math
import lmdb
import numpy as np
from numpy import float32
from tqdm import tqdm

import unicodedata
import librosa
import pyarrow
import pickle
import subprocess

from data_utils import *


def read_subtitle(vid):
    postfix_in_filename = '-en.vtt'
    file_list = glob.glob(my_config.SUBTITLE_PATH + '/*' + vid + postfix_in_filename)
    if len(file_list) > 1:
        print('more than one subtitle. check this.', file_list)
        assert False
    if len(file_list) == 1:
        return WebVTT().read(file_list[0])
    else:
        return []


# turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    
def load_jsons(video_path):
    openposes = []
    json_path = os.path.join(video_path,"keypoints")
    json_paths = sorted(os.listdir(json_path))
    
    exposes = []
    expose_folder = os.path.join(video_path,"expose_keypoints")
    expose_paths = sorted(os.listdir(expose_folder))
    
    assert len(json_paths) == len(expose_paths)
    
    for i in range(len(json_paths)):
        json_small = os.path.join(json_path,json_paths[i])
        with open(json_small,"r") as f:
            frame = json.load(f)["people"][0]
            
        if 'pose_keypoints_2d' in frame:
            # load keypoints of body, left hand, right hand
            openposes_real = frame["pose_keypoints_2d"]
            hands_left = frame["hand_left_keypoints_2d"]
            hands_right = frame["hand_right_keypoints_2d"]
                
            openpose_keypoints = []
            # delete miss matched finger points
            misses = [0,1*3,5*3,9*3,13*3,17*3]
            
            for j in range(0,len(openposes_real),3):
                tmp_keypoints = [openposes_real[j],openposes_real[j+1]]
                openpose_keypoints.append(tmp_keypoints)
                
            for j in range(0,len(hands_left),3):
                if j not in misses:
                    tmp_keypoints = [hands_left[j],hands_left[j+1]]
                    openpose_keypoints.append(tmp_keypoints)
                
            for j in range(0,len(hands_right),3):
                if j not in misses:
                    tmp_keypoints = [hands_right[j],hands_right[j+1]]
                    openpose_keypoints.append(tmp_keypoints)
                    
            proj_joints = np.array(openpose_keypoints)  
            # only perserves upper body
            openpose_keypoints = np.vstack((proj_joints[0], proj_joints[1:3], proj_joints[5],proj_joints[3],proj_joints[6],proj_joints[4],proj_joints[7],proj_joints[40:55], proj_joints[25:40], proj_joints[0]))
            nose_openpose = openpose_keypoints[-1]
        
            expose_small = os.path.join(expose_folder,expose_paths[i],expose_paths[i]+"_params.npz")
            expose_small = np.load(expose_small)
            center = expose_small['center']
            mm = expose_small['focal_length_in_mm']
            px = expose_small['focal_length_in_px']
            
            proj_joints = expose_small['proj_joints']
            proj_joints = proj_joints * px*11/ mm
            nose_proj = proj_joints[55]
            # nose_proj = nose_proj*px*10/mm
            multiply_x = (nose_openpose-center)[0]/nose_proj[0]
            multiply = (nose_openpose-center)[1]/nose_proj[1]
            proj_joints*=multiply*0.88
            proj_joints += center
            
            
            up_joints = np.vstack((proj_joints[9], proj_joints[12], proj_joints[16:22], proj_joints[25:40], proj_joints[40:55], proj_joints[55])) .astype('float32')
            up_joints = up_joints # 43*2

            assert (openpose_keypoints.shape == up_joints.shape)
            
            openpose_keypoints[0] = up_joints[0]
            
            for params in range(len(openpose_keypoints)):
                if openpose_keypoints[params][0] == 0.0 or openpose_keypoints[params][1] == 0.0:
                    openpose_keypoints[params] = up_joints[params]

                    
            
            openposes.append(openpose_keypoints)
            exposes.append(up_joints)

    return openposes,exposes
    
    


# lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)  # isolate some marks
    s = re.sub(r"(['])", r"", s)  # remove apostrophe
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)  # replace other characters with whitespace
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def normalize_subtitle(vtt_subtitle):
    for i, sub in enumerate(vtt_subtitle):
        vtt_subtitle[i].text = normalize_string(vtt_subtitle[i].text)
    return vtt_subtitle

OUTPUT_PATH = "openpose_2d_data_v2"

def make_ted_gesture_dataset():
    dataset_train = []
    dataset_val = []
    dataset_test = []
    n_saved_clips = [0, 0, 0]

    out_lmdb_dir_train = OUTPUT_PATH + '/train'
    out_lmdb_dir_val = OUTPUT_PATH + '/val'
    out_lmdb_dir_test = OUTPUT_PATH + '/test'
    
    if not os.path.exists(out_lmdb_dir_train):
        os.makedirs(out_lmdb_dir_train)
    if not os.path.exists(out_lmdb_dir_val):
        os.makedirs(out_lmdb_dir_val)
    if not os.path.exists(out_lmdb_dir_test):
        os.makedirs(out_lmdb_dir_test)

    all_clips = get_all_clips()
    # video_files = video_files[:10]
    
    for i,clip_file in enumerate(tqdm(all_clips)):
        print(clip_file)
        vid = os.path.split(clip_file)[-1]
        video_path = os.path.join(clip_file,vid+".mp4")
        print(vid)
        real_start_no = int(vid.split("__")[-2])
        
        if not os.path.exists(clip_file + '/' + vid + '.mp3'):
            command = "ffmpeg -i " + clip_file + '/' + vid + '.mp4' + " "+clip_file + '/' + vid + '.mp3'
            print(command)
            subprocess.call(command, shell=True)

        video_wrapper = read_video_simple(clip_file + '/' + vid + '.mp4')
        
        audio_path = clip_file + '/' + vid + '.mp3'
        audio_file, _ = librosa.load(audio_path, sr = 16000, mono = True)
        
        small_cropped = os.path.join(clip_file,"frames_cropped")
        
        dataset_train.append({'vid': vid, 'clips': [], 'framerate': video_wrapper.framerate})
        dataset_val.append({'vid': vid, 'clips': [], 'framerate': video_wrapper.framerate})
        dataset_test.append({'vid': vid, 'clips': [], 'framerate': video_wrapper.framerate})
        
        valid_clip_count = 0
        
        if os.path.exists(small_cropped):
            for small_clip in (os.listdir(small_cropped)):
                start_frame_no = int(small_clip.split("_")[-2])
                end_frame_no = int(small_clip.split("_")[-1])
                
                json_path = os.path.join(small_cropped,small_clip,"keypoints")
                
                expose_folder = os.path.join(small_cropped,small_clip,"expose_keypoints")
                
                if os.path.exists(json_path) and os.path.exists(expose_folder):
                    json_paths = sorted(os.listdir(json_path))
                    expose_paths = sorted(os.listdir(expose_folder))
                    if len(json_paths)==len(expose_paths):
                        
                        clip_skeleton_2d, clip_skeleton_3d = load_jsons(os.path.join(small_cropped,small_clip))
                        start_time = (start_frame_no-real_start_no) / video_wrapper.framerate
                        end_time = (end_frame_no-real_start_no) / video_wrapper.framerate
                        audio_start = math.floor((start_frame_no-real_start_no) / video_wrapper.framerate * 16000)
                        audio_end = math.ceil((end_frame_no-real_start_no) / video_wrapper.framerate * 16000)
                        audio_raw = audio_file[audio_start:audio_end].astype('float16')

                        melspec = librosa.feature.melspectrogram(y=audio_raw, sr=16000, n_fft=1024, hop_length=512, power=2)
                        log_melspec = librosa.power_to_db(melspec, ref=np.max)  # mels x time
                        audio_feat = log_melspec.astype('float16')
                        
                        # train/val/test split
                        if valid_clip_count % 10 == 9:
                            dataset = dataset_test
                            dataset_idx = 2
                        elif valid_clip_count % 10 == 8:
                            dataset = dataset_val
                            dataset_idx = 1
                        else:
                            dataset = dataset_train
                            dataset_idx = 0
                        valid_clip_count += 1
                        
                        # proceed if skeleton list is not empty
                        if len(clip_skeleton_2d) > 0:
                            # save subtitles and skeletons corresponding to clips
                            n_saved_clips[dataset_idx] += 1
                            dataset[-1]['clips'].append({ 'skeletons': clip_skeleton_2d,
                                                        'skeletons_3d': clip_skeleton_3d,
                                                        'audio_feat': audio_feat,
                                                        'audio_raw': audio_raw,
                                                        'start_frame_no': start_frame_no, 
                                                        'end_frame_no': end_frame_no,
                                                        'start_time': start_time,
                                                        'end_time': end_time
                                                        })
                            print('{} ({}, {})'.format(vid, start_frame_no, end_frame_no))
                        else:
                            print('{} ({}, {}) - consecutive missing frames'.format(vid, start_frame_no, end_frame_no))

    print('writing to pickle...')
    with open(out_lmdb_dir_train + 'ted_expressive_dataset_train.pickle', 'wb') as f:
        pickle.dump(dataset_train, f)
    with open(out_lmdb_dir_train + 'ted_expressive_dataset_val.pickle', 'wb') as f:
        pickle.dump(dataset_val, f)
    with open(out_lmdb_dir_train + 'ted_expressive_dataset_test.pickle', 'wb') as f:
        pickle.dump(dataset_test, f)

    map_size = 1024 * 100  # in MB
    map_size <<= 20  # in B
    env_train = lmdb.open(out_lmdb_dir_train, map_size=map_size)
    env_val = lmdb.open(out_lmdb_dir_val, map_size=map_size)
    env_test = lmdb.open(out_lmdb_dir_test, map_size=map_size)

    # lmdb train
    with env_train.begin(write=True) as txn:
        for idx, dic in enumerate(dataset_train):
            k = '{:010}'.format(idx).encode('ascii')
            v = pickle.dumps(dic)
            txn.put(k, v)
    env_train.close()

    # lmdb val
    with env_val.begin(write=True) as txn:
        for idx, dic in enumerate(dataset_val):
            k = '{:010}'.format(idx).encode('ascii')
            v = pickle.dumps(dic)
            txn.put(k, v)
    env_val.close()

    # lmdb test
    with env_test.begin(write=True) as txn:
        for idx, dic in enumerate(dataset_test):
            k = '{:010}'.format(idx).encode('ascii')
            v = pickle.dumps(dic)
            txn.put(k, v)
    env_test.close()
    

    print('no. of saved clips: train {}, val {}, test {}'.format(n_saved_clips[0], n_saved_clips[1], n_saved_clips[2]))
    
def get_all_clips(datapath = '/home/tiger/nfs/workspace/DiffGesture/openpose_dataset_v4'):
    all_smallclips = []
    big_clips = os.listdir(datapath)
    for big_clip in big_clips:
        cropped = os.path.join(datapath, big_clip)
        all_smallclips.append(cropped)
        # if os.path.exists(cropped):
        #     small_clips = os.listdirs(cropped)
        #     for small_clip in small_clips:
        #         full_path = os.path.join(cropped,small_clip)
        #         all_smallclips.append(full_path)
    return all_smallclips
    


if __name__ == '__main__':
    make_ted_gesture_dataset()
