""" create data samples """
import logging
from collections import defaultdict

import lmdb
import math
import numpy as np
import pyarrow
import tqdm
from sklearn.preprocessing import normalize
import pickle
import os
import json

import utils.data_utils_expressive
from data_loader.motion_preprocessor_expressive import MotionPreprocessor


class DataPreprocessor:
    def __init__(self, clip_lmdb_dir, out_lmdb_dir, n_poses, subdivision_stride,
                 pose_resampling_fps, mean_pose, mean_dir_vec, disable_filtering=False):
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.mean_pose = mean_pose
        self.mean_dir_vec = mean_dir_vec
        self.disable_filtering = disable_filtering

        self.src_lmdb_env = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
        with self.src_lmdb_env.begin() as txn:
            self.n_videos = txn.stat()['entries']

        self.spectrogram_sample_length = utils.data_utils_expressive.calc_spectrogram_length_from_motion_length(self.n_poses, self.skeleton_resampling_fps)
        self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * 16000)

        # create db for samples
        map_size = 1024 * 200  # in MB
        map_size <<= 20  # in B
        self.dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        self.n_out_samples = 0

    def run(self):
        n_filtered_out = defaultdict(int)
        src_txn = self.src_lmdb_env.begin(write=False)

        # sampling and normalization
        cursor = src_txn.cursor()
        for key, value in cursor:
            video = pickle.loads(value)
            vid = video['vid']
            clips = video['clips']
            for clip_idx, clip in enumerate(clips):
                filtered_result = self._sample_from_clip(vid, clip)
                for type in filtered_result.keys():
                    n_filtered_out[type] += filtered_result[type]

        # print stats
        with self.dst_lmdb_env.begin() as txn:
            print('no. of samples: ', txn.stat()['entries'])
            n_total_filtered = 0
            for type, n_filtered in n_filtered_out.items():
                print('{}: {}'.format(type, n_filtered))
                n_total_filtered += n_filtered
            print('no. of excluded samples: {} ({:.1f}%)'.format(
                n_total_filtered, 100 * n_total_filtered / (txn.stat()['entries'] + n_total_filtered)))

        # close db
        self.src_lmdb_env.close()
        self.dst_lmdb_env.sync()
        self.dst_lmdb_env.close()
        # assert False

    def _sample_from_clip(self, vid, clip):
        clip_skeleton = clip['skeletons_3d']
        clip_openpose_skeleton = clip['skeletons']
        print(clip_skeleton[0])
        clip_audio = clip['audio_feat']
        clip_audio_raw = clip['audio_raw']
        # clip_word_list = clip['words']
        clip_s_f, clip_e_f = clip['start_frame_no'], clip['end_frame_no']
        clip_s_t, clip_e_t = clip['start_time'], clip['end_time']

        n_filtered_out = defaultdict(int)

        # skeleton resampling
        clip_skeleton = utils.data_utils_expressive.resample_pose_seq(clip_skeleton, clip_e_t - clip_s_t, self.skeleton_resampling_fps)
        clip_openpose_skeleton = utils.data_utils_expressive.resample_pose_seq(clip_openpose_skeleton, clip_e_t - clip_s_t, self.skeleton_resampling_fps)
        
        # divide
        aux_info = []
        sample_skeletons_list = []
        sample_audio_list = []
        sample_spectrogram_list = []
        sample_openpose_list = []

        num_subdivision = math.floor(
            (len(clip_skeleton) - self.n_poses)
            / self.subdivision_stride) + 1  # floor((K - (N+M)) / S) + 1
        expected_audio_length = utils.data_utils_expressive.calc_spectrogram_length_from_motion_length(len(clip_skeleton), self.skeleton_resampling_fps)
        assert abs(expected_audio_length - clip_audio.shape[1]) <= 5, 'audio and skeleton lengths are different'

        for i in range(num_subdivision):
            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses

            sample_skeletons = clip_skeleton[start_idx:fin_idx]
            sample_openpose_skeletons = clip_openpose_skeleton[start_idx:fin_idx]
            subdivision_start_time = clip_s_t + start_idx / self.skeleton_resampling_fps
            subdivision_end_time = clip_s_t + fin_idx / self.skeleton_resampling_fps
            # sample_words = self.get_words_in_time_range(word_list=clip_word_list,
            #                                             start_time=subdivision_start_time,
            #                                             end_time=subdivision_end_time)

            # spectrogram
            audio_start = math.floor(start_idx / len(clip_skeleton) * clip_audio.shape[1])
            audio_end = audio_start + self.spectrogram_sample_length
            if audio_end > clip_audio.shape[1]:  # correct size mismatch between poses and audio
                # logging.info('expanding audio array, audio start={}, end={}, clip_length={}'.format(
                #     audio_start, audio_end, clip_audio.shape[1]))
                n_padding = audio_end - clip_audio.shape[1]
                padded_data = np.pad(clip_audio, ((0, 0), (0, n_padding)), mode='symmetric')
                sample_spectrogram = padded_data[:, audio_start:audio_end]
            else:
                sample_spectrogram = clip_audio[:, audio_start:audio_end]

            # raw audio
            audio_start = math.floor(start_idx / len(clip_skeleton) * len(clip_audio_raw))
            audio_end = audio_start + self.audio_sample_length
            if audio_end > len(clip_audio_raw):  # correct size mismatch between poses and audio
                # logging.info('expanding audio array, audio start={}, end={}, clip_length={}'.format(
                #     audio_start, audio_end, len(clip_audio_raw)))
                n_padding = audio_end - len(clip_audio_raw)
                padded_data = np.pad(clip_audio_raw, (0, n_padding), mode='symmetric')
                sample_audio = padded_data[audio_start:audio_end]
            else:
                sample_audio = clip_audio_raw[audio_start:audio_end]

            if True:
                # filtering motion skeleton data
                sample_skeletons = sample_skeletons.reshape(sample_skeletons.shape[0],-1)
                # sample_skeletons, filtering_message = MotionPreprocessor(sample_skeletons, self.mean_pose).get()
                filtering_message = True
                is_correct_motion = (sample_skeletons != [])
                motion_info = {'vid': vid,
                               'start_frame_no': clip_s_f + start_idx,
                               'end_frame_no': clip_s_f + fin_idx,
                               'start_time': subdivision_start_time,
                               'end_time': subdivision_end_time,
                               'is_correct_motion': is_correct_motion, 'filtering_message': filtering_message}

                if (is_correct_motion or self.disable_filtering):
                    print(f"vid: {vid}, start_frame: {clip_s_f + start_idx}, end_frame: {clip_s_f + fin_idx}")
                   
                    sample_skeletons_list.append(sample_skeletons)
                    sample_openpose_list.append(sample_openpose_skeletons)
                    sample_audio_list.append(sample_audio)
                    sample_spectrogram_list.append(sample_spectrogram)
                    aux_info.append(motion_info)
                    
                else:
                    n_filtered_out[filtering_message] += 1

        if len(sample_skeletons_list) > 0:
            with self.dst_lmdb_env.begin(write=True) as txn:
                for poses,openposes, audio, spectrogram, aux in zip(sample_skeletons_list, sample_openpose_list,
                                                                 sample_audio_list, sample_spectrogram_list,
                                                                 aux_info):
                    # preprocessing for poses
                    # poses = np.asarray(poses)
                    # dir_vec = utils.data_utils_expressive.convert_pose_seq_to_dir_vec(poses)
                    # normalized_dir_vec = self.normalize_dir_vec(dir_vec, self.mean_dir_vec)
                    
                    openposes = np.asarray(openposes)
                    dir_vec_openpose = utils.data_utils_expressive.convert_pose_seq_to_dir_vec_2d(openposes)
                    # normalized_dir_vec_openpose = self.normalize_dir_vec(dir_vec, self.mean_dir_vec)
                    
                    # print("start to save media")
                    # mediapipe_poses = np.asarray(mediapipe_poses)
                    # dir_vec_media = utils.data_utils_expressive.convert_pose_seq_to_dir_vec_mediapipe_2d(mediapipe_poses)
                    # print(dir_vec_media.shape)
                    # normalized_dir_vec = self.normalize_dir_vec(dir_vec, self.mean_dir_vec)

                    # save
                    k = '{:010}'.format(self.n_out_samples).encode('ascii')
                    v = [openposes,dir_vec_openpose, audio, spectrogram, aux]
                    v = pickle.dumps(v)
                    txn.put(k, v)
                    self.n_out_samples += 1

        return n_filtered_out
    
    def load_mediapipe(self,vid,start_frame,end_frame):
        source_path = "/home/tiger/nfs/ted_expressive/mediapipe_keypoints"
        vid_keypoints_path = os.path.join(source_path,vid+"_")
        keypoints = sorted(os.listdir(vid_keypoints_path))
        mediapipe_kepoints = []
        for i in range(start_frame,end_frame):
            # print(f"open {os.path.join(vid_keypoints_path,keypoints[i])}")
            with open(os.path.join(vid_keypoints_path,keypoints[i])) as f:
              media_keypoint = json.load(f)
              if(len(media_keypoint["keypoints_3d_poses"])<=0):
                  with open(os.path.join(vid_keypoints_path,keypoints[i+2])) as f1:
                      media_keypoint = json.load(f1)
                      body,left,right = load_normalized_points(media_keypoint)
              else:
                  body,left,right = load_normalized_points(media_keypoint)
            # if body == None:
            #     media_keypoint = json.load(os.path.join(vid_keypoints_path,keypoints[i-1]))
            #     body,left,right = load_normalized_points(media_keypoint)
            # full_body = np.concatenate((body,left,right),0)
            full_body = body
            mediapipe_kepoints.append(full_body)
        
        return mediapipe_kepoints
        
        

    @staticmethod
    def normalize_dir_vec(dir_vec, mean_dir_vec):
        return dir_vec - mean_dir_vec

    @staticmethod
    def get_words_in_time_range(word_list, start_time, end_time):
        words = []

        for word in word_list:
            _, word_s, word_e = word[0], word[1], word[2]

            if word_s >= end_time:
                break

            if word_e <= start_time:
                continue

            words.append(word)

        return words

    @staticmethod
    def unnormalize_data(normalized_data, data_mean, data_std, dimensions_to_ignore):
        """
        this method is from https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12
        """
        T = normalized_data.shape[0]
        D = data_mean.shape[0]

        origData = np.zeros((T, D), dtype=np.float32)
        dimensions_to_use = []
        for i in range(D):
            if i in dimensions_to_ignore:
                continue
            dimensions_to_use.append(i)
        dimensions_to_use = np.array(dimensions_to_use)

        origData[:, dimensions_to_use] = normalized_data

        # potentially inefficient, but only done once per experiment
        stdMat = data_std.reshape((1, D))
        stdMat = np.repeat(stdMat, T, axis=0)
        meanMat = data_mean.reshape((1, D))
        meanMat = np.repeat(meanMat, T, axis=0)
        origData = np.multiply(origData, stdMat) + meanMat

        return origData

import time
def load_normalized_points(dict1):
        width = 1280
        height = 720
        body_keypoints = dict1["keypoints_3d_poses"]
        left_hand = dict1["keypoints_3d_left_hands"]
        right_hand = dict1["keypoints_3d_right_hands"]
        length = len(body_keypoints)
        missed = 0
        for i in range(len(body_keypoints)):
            if body_keypoints[i] == 0:
                assert False
            # assert len(left_hand) > 0 and len(right_hand)>0
        
        # print(len(body_keypoints),len(left_hand),len(right_hand))
        # calculate shoulder center
        should_centerx = (body_keypoints[3+0]+body_keypoints[0])/2
        should_centery = (body_keypoints[3+1]+body_keypoints[1])/2
        should_centerz = (body_keypoints[3+2]+body_keypoints[2])/2
        # normalize body keypoints
        for i in range(0,(len(body_keypoints)),3):
            body_keypoints[i] = (body_keypoints[i]-should_centerx)/width
            body_keypoints[i+1] = (body_keypoints[i+1]-should_centery)/height
            body_keypoints[i+2] = (body_keypoints[i+2]-should_centerz)/width
            
        # left_wristz = body_keypoints[4*3+2]
        # right_wristz = body_keypoints[5*3+2]
        
        # # normalize left hand points
        # for i in range(len(left_hand),3):
        #     left_hand[i] = (left_hand[i]-should_centerx)/width
        #     left_hand[i+1] = (left_hand[i+1]-should_centery)/height
        #     left_hand[i+2] = left_hand[i+2]/width+left_wristz
            
        # # normalize right hand points
        # for i in range(len(right_hand),3):
        #     right_hand[i] = (right_hand[i]-should_centerx)/width
        #     right_hand[i+1] = (right_hand[i+1]-should_centery)/height
        #     right_hand[i+2] = right_hand[i+2]/width+right_wristz
            
        body_keypoints = np.array(body_keypoints)
        # left_hand = np.array(left_hand)
        # right_hand = np.array(right_hand)
            
        body_keypoints = body_keypoints.reshape(-1,3)[:,:-1].view()
        # left_hand = left_hand.reshape(-1,3)[:,:-1].view()
        # right_hand = right_hand.reshape(-1,3)[:,:-1].view()
        
        
        return body_keypoints, left_hand, right_hand
            