import re

import librosa
import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dir_vec_pairs = [
    (0, 1, 0.26), # 0, spine-neck
    (1, 2, 0.22), # 1, neck-left shoulder
    (1, 3, 0.22), # 2, neck-right shoulder
    (2, 4, 0.36), # 3, left shoulder-elbow
    (4, 6, 0.33), # 4, left elbow-wrist

    (6, 8, 0.137), # 5 wrist-left index 1
    (8, 9, 0.044), # 6
    (9, 10, 0.031), # 7

    (6, 11, 0.144), # 8 wrist-left middle 1
    (11, 12, 0.042), # 9
    (12, 13, 0.033), # 10

    (6, 14, 0.127), # 11 wrist-left pinky 1
    (14, 15, 0.027), # 12
    (15, 16, 0.026), # 13

    (6, 17, 0.134), # 14 wrist-left ring 1
    (17, 18, 0.039), # 15
    (18, 19, 0.033), # 16

    (6, 20, 0.068), # 17 wrist-left thumb 1
    (20, 21, 0.042), # 18
    (21, 22, 0.036), # 19

    (3, 5, 0.36), # 20, right shoulder-elbow
    (5, 7, 0.33), # 21, right elbow-wrist

    (7, 23, 0.137), # 22 wrist-right index 1
    (23, 24, 0.044), # 23
    (24, 25, 0.031), # 24

    (7, 26, 0.144), # 25 wrist-right middle 1
    (26, 27, 0.042), # 26
    (27, 28, 0.033), # 27

    (7, 29, 0.127), # 28 wrist-right pinky 1
    (29, 30, 0.027), # 29
    (30, 31, 0.026), # 30

    (7, 32, 0.134), # 31 wrist-right ring 1
    (32, 33, 0.039), # 32
    (33, 34, 0.033), # 33

    (7, 35, 0.068), # 34 wrist-right thumb 1
    (35, 36, 0.042), # 35
    (36, 37, 0.036), # 36

    (1, 38, 0.18), # 37, neck-nose
    (38, 39, 0.14), # 38, nose-right eye
    (38, 40, 0.14), # 39, nose-left eye
    (39, 41, 0.15), # 40, right eye-right ear
    (40, 42, 0.15), # 41, left eye-left ear
]

dir_vec_pairs_fixed = [
    (0, 1, 0.25), # 0, spine-neck
    (1, 2, 0.25), # 1, neck-left shoulder
    (1, 3, 0.25), # 2, neck-right shoulder
    (2, 4, 0.25), # 3, left shoulder-elbow
    (4, 6, 0.25), # 4, left elbow-wrist

    (6, 8, 0.1), # 5 wrist-left index 1
    (8, 9, 0.03), # 6
    (9, 10, 0.03), # 7

    (6, 11, 0.1), # 8 wrist-left middle 1
    (11, 12, 0.03), # 9
    (12, 13, 0.03), # 10

    (6, 14, 0.1), # 11 wrist-left pinky 1
    (14, 15, 0.03), # 12
    (15, 16, 0.03), # 13

    (6, 17, 0.1), # 14 wrist-left ring 1
    (17, 18, 0.03), # 15
    (18, 19, 0.03), # 16

    (6, 20, 0.1), # 17 wrist-left thumb 1
    (20, 21, 0.03), # 18
    (21, 22, 0.03), # 19

    (3, 5, 0.2), # 20, right shoulder-elbow
    (5, 7, 0.2), # 21, right elbow-wrist

    (7, 23, 0.1), # 22 wrist-right index 1
    (23, 24, 0.03), # 23
    (24, 25, 0.03), # 24

    (7, 26, 0.1), # 25 wrist-right middle 1
    (26, 27, 0.03), # 26
    (27, 28, 0.03), # 27

    (7, 29, 0.1), # 28 wrist-right pinky 1
    (29, 30, 0.03), # 29
    (30, 31, 0.03), # 30

    (7, 32, 0.1), # 31 wrist-right ring 1
    (32, 33, 0.03), # 32
    (33, 34, 0.03), # 33

    (7, 35, 0.1), # 34 wrist-right thumb 1
    (35, 36, 0.03), # 35
    (36, 37, 0.03), # 36

    (1, 38, 0.2), # 37, neck-nose
    (38, 39, 0.17), # 38, nose-right eye
    (38, 40, 0.17), # 39, nose-left eye
    (39, 41, 0.17), # 40, right eye-right ear
    (40, 42, 0.17), # 41, left eye-left ear
]

dir_vec_pairs_media = [
    (0, 1, 0.35), # 0, left-right shoulder
    (0, 2, 0.32), # 1, left arm
    (1, 3, 0.32), # 2, right arm
    (2, 4, 0.32), # 3, left arm 2
    (3, 5, 0.32), # 4, right arm 2

    (0, 6, 0.34), # 5 left body
    (1, 7, 0.34), # 6 right body
    (6, 7, 0.20), # 7 lower body

    # (8, 9, 0.044), # 8 wrist-left middle 1
    # (9, 10, 0.042), # 9
    # (10, 11, 0.033), # 10
    # (11, 12, 0.033),
    
    # (8, 13, 0.084), # left hand
    # (13, 14, 0.042),
    # (14, 15, 0.042),
    # (15, 16, 0.042),
    
    # (13, 17, 0.033),
    # (17, 18, 0.042),
    # (18, 19, 0.042),
    # (19, 20, 0.042),
    
    # (17, 21, 0.033),
    # (21, 22, 0.042),
    # (22, 23, 0.042),
    # (23, 24, 0.042),
    
    # (21, 25, 0.033),
    # (25, 26, 0.042),
    # (26, 27, 0.042),
    # (27, 28, 0.042),
    
    # (8+21, 13+21, 0.084), # right hand
    # (13+21, 14+21, 0.042),
    # (14+21, 15+21, 0.042),
    # (15+21, 16+21, 0.042),
    
    # (13+21, 17+21, 0.033),
    # (17+21, 18+21, 0.042),
    # (18+21, 19+21, 0.042),
    # (19+21, 20+21, 0.042),
    
    # (17+21, 21+21, 0.033),
    # (21+21, 22+21, 0.042),
    # (22+21, 23+21, 0.042),
    # (23+21, 24+21, 0.042),
    
    # (21+21, 25+21, 0.033),
    # (25+21, 26+21, 0.042),
    # (26+21, 27+21, 0.042),
    # (27+21, 28+21, 0.042),
    
]

dir_vec_pairs_media_hands = [
    (0, 1, 0.35), # 0, left-right shoulder
    (0, 2, 0.32), # 1, left arm
    (1, 3, 0.32), # 2, right arm
    (2, 4, 0.32), # 3, left arm 2
    (3, 5, 0.32), # 4, right arm 2

    (0, 6, 0.34), # 5 left body
    (1, 7, 0.34), # 6 right body
    (6, 7, 0.20), # 7 lower body

    (8, 9, 0.044), # 8 wrist-left middle 1
    (9, 10, 0.042), # 9
    (10, 11, 0.033), # 10
    (11, 12, 0.033),
    
    (8, 13, 0.084), # left hand
    (13, 14, 0.042),
    (14, 15, 0.042),
    (15, 16, 0.042),
    
    (13, 17, 0.033),
    (17, 18, 0.042),
    (18, 19, 0.042),
    (19, 20, 0.042),
    
    (17, 21, 0.033),
    (21, 22, 0.042),
    (22, 23, 0.042),
    (23, 24, 0.042),
    
    (21, 25, 0.033),
    (25, 26, 0.042),
    (26, 27, 0.042),
    (27, 28, 0.042),
    
    (8+21, 13+21, 0.084), # right hand
    (13+21, 14+21, 0.042),
    (14+21, 15+21, 0.042),
    (15+21, 16+21, 0.042),
    
    (13+21, 17+21, 0.033),
    (17+21, 18+21, 0.042),
    (18+21, 19+21, 0.042),
    (19+21, 20+21, 0.042),
    
    (17+21, 21+21, 0.033),
    (21+21, 22+21, 0.042),
    (22+21, 23+21, 0.042),
    (23+21, 24+21, 0.042),
    
    (21+21, 25+21, 0.033),
    (25+21, 26+21, 0.042),
    (26+21, 27+21, 0.042),
    (27+21, 28+21, 0.042),
    
]

def normalize_string(s):
    """ lowercase, trim, and remove non-letter characters """
    s = s.lower().strip()
    s = re.sub(r"([,.!?])", r" \1 ", s)  # isolate some marks
    s = re.sub(r"(['])", r"", s)  # remove apostrophe
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)  # replace other characters with whitespace
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def remove_tags_marks(text):
    reg_expr = re.compile('<.*?>|[.,:;!?]+')
    clean_text = re.sub(reg_expr, '', text)
    return clean_text


def extract_melspectrogram(y, sr=16000):
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, power=2)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)  # mels x time
    log_melspec = log_melspec.astype('float16')
    return log_melspec


def calc_spectrogram_length_from_motion_length(n_frames, fps):
    ret = (n_frames / fps * 16000 - 1024) / 512 + 1
    return int(round(ret))


def resample_pose_seq(poses, duration_in_sec, fps):
    n = len(poses)
    x = np.arange(0, n)
    y = poses
    f = interp1d(x, y, axis=0, kind='linear', fill_value='extrapolate')
    expected_n = duration_in_sec * fps
    x_new = np.arange(0, n, n / expected_n)
    interpolated_y = f(x_new)
    if hasattr(poses, 'dtype'):
        interpolated_y = interpolated_y.astype(poses.dtype)
    return interpolated_y


def time_stretch_for_words(words, start_time, speech_speed_rate):
    for i in range(len(words)):
        if words[i][1] > start_time:
            words[i][1] = start_time + (words[i][1] - start_time) / speech_speed_rate
        words[i][2] = start_time + (words[i][2] - start_time) / speech_speed_rate

    return words


def make_audio_fixed_length(audio, expected_audio_length):
    if len(audio.shape) == 1:
        n_padding = expected_audio_length - len(audio)
        if n_padding > 0:
            audio = np.pad(audio, (0, n_padding), mode='symmetric')
        else:
            audio = audio[0:expected_audio_length]
    else:
        n_padding = expected_audio_length - len(audio)
        if n_padding > 0:
            audio = np.pad(audio, (0, n_padding), mode='symmetric')
        else:
            audio = audio[0:expected_audio_length,:]
    return audio


def convert_dir_vec_to_pose(vec):
    # vec = np.array(vec)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((43, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[pair[1]] = joint_pos[pair[0]] + [pair[2]] * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 43, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + [pair[2]] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 42, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 43, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + [pair[2]] * vec[:, :, j]
    else:
        assert False

    return joint_pos

def convert_dir_vec_to_pose_l2n(vec):

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((43, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[pair[1]] = joint_pos[pair[0]] + [pair[2]] * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 43, 3))
        for i in range(vec.shape[1]):
            vec[:, i, :] = normalize(vec[:, i, :], axis=1)
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + [pair[2]] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 42, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 43, 3))
        
        for j in range(vec.shape[0]):  # batch
            for i in range(len(dir_vec_pairs)):
                vec[j, :, i, :] = normalize(vec[j, :, i, :], axis=1)  # to unit length
        
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + [pair[2]] * vec[:, :, j]
    else:
        assert False

    return joint_pos

def convert_dir_vec_to_norm_vec(vec):

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 3:
        for i in range(vec.shape[1]):
            vec[:, i, :] = normalize(vec[:, i, :], axis=1)
            
    elif len(vec.shape) == 4:  # (batch, seq, 42, 3)
        for j in range(vec.shape[0]):  # batch
            for i in range(len(dir_vec_pairs)):
                vec[j, :, i, :] = normalize(vec[j, :, i, :], axis=1)  # to unit length
    else:
        assert False

    return vec

def convert_dir_vec_to_pose_2d(vec):
    # vec = np.array(vec)

    if vec.shape[-1] != 2:
        vec = vec.reshape(vec.shape[:-1] + (-1, 2))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((39, 2))
        for j, pair in enumerate(dir_vec_pairs[:-4]):
            joint_pos[pair[1]] = joint_pos[pair[0]] + [pair[2]] * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 39,2))
        for j, pair in enumerate(dir_vec_pairs[:-4]):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + [pair[2]] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 42, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 39, 2))
        for j, pair in enumerate(dir_vec_pairs[:-4]):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + [pair[2]] * vec[:, :, j]
    else:
        assert False

    return joint_pos

def convert_dir_vec_to_pose_fixed(vec):
    # vec = np.array(vec)
    multiple = 1.2

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((43, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[pair[1]] = joint_pos[pair[0]] + multiple * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 43, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + multiple * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 42, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 43, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + multiple * vec[:, :, j]
    else:
        assert False
        
    

    return joint_pos

def convert_dir_vec_to_pose_real(vec,start_joint):
    # vec = np.array(vec)
    multiple = 1

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((43, 3))+start_joint
        print(joint_pos)
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[pair[1]] = joint_pos[pair[0]] + multiple * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 43, 3))+start_joint
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + multiple * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 42, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 43, 3))+start_joint
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + multiple * vec[:, :, j]
    else:
        assert False
        
    

    return joint_pos

def convert_dir_vec_to_pose_mediapipe(vec):
    # vec = np.array(vec)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((8, 3))
        for j, pair in enumerate(dir_vec_pairs_media):
            joint_pos[pair[1]] = joint_pos[pair[0]] + [pair[2]] * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 8, 3))
        for j, pair in enumerate(dir_vec_pairs_media):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + [pair[2]] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 42, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 8, 3))
        for j, pair in enumerate(dir_vec_pairs_media):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + [pair[2]] * vec[:, :, j]
    else:
        assert False
        
    print(joint_pos.shape)
        
    for i in range(len(joint_pos)):
        centerx = (joint_pos[i,0,0]+joint_pos[i,1,0])/2
        centery = (joint_pos[i,0,1]+joint_pos[i,1,1])/2
        centerz = (joint_pos[i,0,2]+joint_pos[i,1,2])/2
        
        for j in range(0,len(joint_pos[i])):
            joint_pos[i][j][0] = joint_pos[i][j][0]-centerx
            joint_pos[i][j][1] = joint_pos[i][j][1]-centery
            joint_pos[i][j][2] = joint_pos[i][j][2]-centerz

    return joint_pos

def convert_dir_vec_to_pose_mediapipe_2d(vec):
    # vec = np.array(vec)

    if vec.shape[-1] != 2:
        vec = vec.reshape(vec.shape[:-1] + (-1, 2))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((8, 2))
        for j, pair in enumerate(dir_vec_pairs_media):
            joint_pos[pair[1]] = joint_pos[pair[0]] + [pair[2]] * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 8, 2))
        for j, pair in enumerate(dir_vec_pairs_media):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + [pair[2]] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 42, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 8, 2))
        for j, pair in enumerate(dir_vec_pairs_media):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + [pair[2]] * vec[:, :, j]
    else:
        assert False
        
    print(joint_pos.shape)
        
    for i in range(len(joint_pos)):
        centerx = (joint_pos[i,0,0]+joint_pos[i,1,0])/2
        centery = (joint_pos[i,0,1]+joint_pos[i,1,1])/2
        
        for j in range(0,len(joint_pos[i])):
            joint_pos[i][j][0] = joint_pos[i][j][0]-centerx
            joint_pos[i][j][1] = joint_pos[i][j][1]-centery

    return joint_pos

def convert_dir_vec_to_pose_mediapipe_3d_to2d(vec):
    # vec = np.array(vec)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((8, 2))
        for j, pair in enumerate(dir_vec_pairs_media):
            joint_pos[pair[1]] = joint_pos[pair[0]] + [pair[2]] * vec[j]
    elif len(vec.shape) == 3:
        vec = vec[:,:,:-1]
        print(vec.shape,"!!")
        joint_pos = np.zeros((vec.shape[0], 8, 2))
        for j, pair in enumerate(dir_vec_pairs_media):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + [pair[2]] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 42, 3)
        vec = np.concatenate((vec[:,:,:,0],vec[:,:,:,2]),3)
        print(vec.shape,"!!")
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 8, 2))
        for j, pair in enumerate(dir_vec_pairs_media):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + [pair[2]] * vec[:, :, j]
    else:
        assert False
        
    print(joint_pos.shape)
        
    for i in range(len(joint_pos)):
        centerx = (joint_pos[i,0,0]+joint_pos[i,1,0])/2
        centery = (joint_pos[i,0,1]+joint_pos[i,1,1])/2
        
        for j in range(0,len(joint_pos[i])):
            joint_pos[i][j][0] = joint_pos[i][j][0]-centerx
            joint_pos[i][j][1] = joint_pos[i][j][1]-centery

    return joint_pos

def convert_dir_vec_to_pose_mediapipe_hands(vec):
    # vec = np.array(vec)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((8, 3))
        for j, pair in enumerate(dir_vec_pairs_media_hands):
            joint_pos[pair[1]] = joint_pos[pair[0]] + [pair[2]] * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 8, 3))
        for j, pair in enumerate(dir_vec_pairs_media_hands):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + [pair[2]] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 42, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 8, 3))
        for j, pair in enumerate(dir_vec_pairs_media_hands):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + [pair[2]] * vec[:, :, j]
    else:
        assert False
        
    print(joint_pos.shape)
        
    for i in range(len(joint_pos)):
        centerx = (joint_pos[i,0,0]+joint_pos[i,1,0])/2
        centery = (joint_pos[i,0,1]+joint_pos[i,1,1])/2
        centerz = (joint_pos[i,0,2]+joint_pos[i,1,2])/2
        
        for j in range(0,len(joint_pos[i])):
            joint_pos[i][j][0] = joint_pos[i][j][0]-centerx
            joint_pos[i][j][1] = joint_pos[i][j][1]-centery
            joint_pos[i][j][2] = joint_pos[i][j][2]-centerz

    return joint_pos

def convert_dir_vec_to_pose_mediapipe_hands_2d(vec):
    # vec = np.array(vec)

    if vec.shape[-1] != 2:
        vec = vec.reshape(vec.shape[:-1] + (-1, 2))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((8, 2))
        for j, pair in enumerate(dir_vec_pairs_media_hands):
            joint_pos[pair[1]] = joint_pos[pair[0]] + [pair[2]] * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 8, 2))
        for j, pair in enumerate(dir_vec_pairs_media_hands):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + [pair[2]] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 42, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 8, 2))
        for j, pair in enumerate(dir_vec_pairs_media_hands):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + [pair[2]] * vec[:, :, j]
    else:
        assert False
        
    print(joint_pos.shape)
        
    for i in range(len(joint_pos)):
        centerx = (joint_pos[i,0,0]+joint_pos[i,1,0])/2
        centery = (joint_pos[i,0,1]+joint_pos[i,1,1])/2
        
        for j in range(0,len(joint_pos[i])):
            joint_pos[i][j][0] = joint_pos[i][j][0]-centerx
            joint_pos[i][j][1] = joint_pos[i][j][1]-centery

    return joint_pos

def convert_dir_vec_to_pose_mediapipe_hands_3d_to2d(vec):
    # vec = np.array(vec)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))
        
    if len(vec.shape) == 2:
        vec = vec[:,:-1]
        joint_pos = np.zeros((8, 2))
        for j, pair in enumerate(dir_vec_pairs_media_hands):
            joint_pos[pair[1]] = joint_pos[pair[0]] + [pair[2]] * vec[j]
    elif len(vec.shape) == 3:
        vec = vec[:,:,:-1]
        joint_pos = np.zeros((vec.shape[0], 8, 2))
        for j, pair in enumerate(dir_vec_pairs_media_hands):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + [pair[2]] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 42, 3)\
        vec = vec[:,:,:,:-1]
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 8, 2))
        for j, pair in enumerate(dir_vec_pairs_media_hands):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + [pair[2]] * vec[:, :, j]
    else:
        assert False
        
    print(joint_pos.shape)
        
    for i in range(len(joint_pos)):
        centerx = (joint_pos[i,0,0]+joint_pos[i,1,0])/2
        centery = (joint_pos[i,0,1]+joint_pos[i,1,1])/2
        
        for j in range(0,len(joint_pos[i])):
            joint_pos[i][j][0] = joint_pos[i][j][0]-centerx
            joint_pos[i][j][1] = joint_pos[i][j][1]-centery

    return joint_pos

def convert_pose_seq_to_dir_vec(pose,norm=True):
    if pose.shape[-1] != 3:
        pose = pose.reshape(pose.shape[:-1] + (-1, 3))

    if len(pose.shape) == 3:
        dir_vec = np.zeros((pose.shape[0], len(dir_vec_pairs), 3))
        for i, pair in enumerate(dir_vec_pairs):
            dir_vec[:, i] = pose[:, pair[1]] - pose[:, pair[0]]
            if norm:
               dir_vec[:, i, :] = normalize(dir_vec[:, i, :], axis=1)  # to unit length
    elif len(pose.shape) == 4:  # (batch, seq, ...)
        dir_vec = np.zeros((pose.shape[0], pose.shape[1], len(dir_vec_pairs), 3))
        for i, pair in enumerate(dir_vec_pairs):
            dir_vec[:, :, i] = pose[:, :, pair[1]] - pose[:, :, pair[0]]
        if norm:
            for j in range(dir_vec.shape[0]):  # batch
                for i in range(len(dir_vec_pairs)):
                    dir_vec[j, :, i, :] = normalize(dir_vec[j, :, i, :], axis=1)  # to unit length
    else:
        assert False

    return dir_vec

def convert_pose_seq_to_dir_vec_2d(pose,norm=True,scale=0.01):
    # dir_vec_pairs = dir_vec_pairs[:-4] # exclue eyes and ears
    print(pose.shape)
    if pose.shape[-1] != 2:
        pose = pose.reshape(pose.shape[:-1] + (-1, 2))

    if len(pose.shape) == 3:
        dir_vec = np.zeros((pose.shape[0], len(dir_vec_pairs)-4, 2))
        for i, pair in enumerate(dir_vec_pairs[:-4]):
            dir_vec[:, i] = pose[:, pair[1]] - pose[:, pair[0]]
            if norm:
               dir_vec[:, i, :] = normalize(dir_vec[:, i, :], axis=1)  # to unit length
            else:
                dir_vec[:, i] = dir_vec[:, i]*scale # multiple scale to reduce number scale
    elif len(pose.shape) == 4:  # (batch, seq, ...)
        dir_vec = np.zeros((pose.shape[0], pose.shape[1], len(dir_vec_pairs)-4, 2))
        for i, pair in enumerate(dir_vec_pairs[:-4]):
            dir_vec[:, :, i] = pose[:, :, pair[1]] - pose[:, :, pair[0]]
        if norm:
            for j in range(dir_vec.shape[0]):  # batch
                for i in range(len(dir_vec_pairs)-4):
                    dir_vec[j, :, i, :] = normalize(dir_vec[j, :, i, :], axis=1)  # to unit length
        else:
            for j in range(dir_vec.shape[0]):  # batch
                for i in range(len(dir_vec_pairs)-4):
                    dir_vec[j, :, i, :] = dir_vec[j, :, i, :]*scale  # multiple scale to reduce number scale
    else:
        assert False

    return dir_vec

def convert_pose_seq_to_dir_vec_mediapipe(pose):
    if pose.shape[-1] != 3:
        pose = pose.reshape(pose.shape[:-1] + (-1, 3))

    if len(pose.shape) == 3:
        dir_vec = np.zeros((pose.shape[0], len(dir_vec_pairs_media), 3))
        for i, pair in enumerate(dir_vec_pairs_media):
            dir_vec[:, i] = pose[:, pair[1]] - pose[:, pair[0]]
            dir_vec[:, i, :] = normalize(dir_vec[:, i, :], axis=1)  # to unit length
    elif len(pose.shape) == 4:  # (batch, seq, ...)
        dir_vec = np.zeros((pose.shape[0], pose.shape[1], len(dir_vec_pairs_media), 3))
        for i, pair in enumerate(dir_vec_pairs_media):
            dir_vec[:, :, i] = pose[:, :, pair[1]] - pose[:, :, pair[0]]
        for j in range(dir_vec.shape[0]):  # batch
            for i in range(len(dir_vec_pairs)):
                dir_vec[j, :, i, :] = normalize(dir_vec[j, :, i, :], axis=1)  # to unit length
    else:
        assert False

    return dir_vec

def convert_pose_seq_to_dir_vec_mediapipe_hands(pose):
    if pose.shape[-1] != 3:
        pose = pose.reshape(pose.shape[:-1] + (-1, 3))

    if len(pose.shape) == 3:
        dir_vec = np.zeros((pose.shape[0], len(dir_vec_pairs_media_hands), 3))
        for i, pair in enumerate(dir_vec_pairs_media_hands):
            dir_vec[:, i] = pose[:, pair[1]] - pose[:, pair[0]]
            dir_vec[:, i, :] = normalize(dir_vec[:, i, :], axis=1)  # to unit length
    elif len(pose.shape) == 4:  # (batch, seq, ...)
        dir_vec = np.zeros((pose.shape[0], pose.shape[1], len(dir_vec_pairs_media_hands), 3))
        for i, pair in enumerate(dir_vec_pairs_media_hands):
            dir_vec[:, :, i] = pose[:, :, pair[1]] - pose[:, :, pair[0]]
        for j in range(dir_vec.shape[0]):  # batch
            for i in range(len(dir_vec_pairs)):
                dir_vec[j, :, i, :] = normalize(dir_vec[j, :, i, :], axis=1)  # to unit length
    else:
        assert False

    return dir_vec

def convert_pose_seq_to_dir_vec_mediapipe_2d(pose):
    if pose.shape[-1] != 2:
        pose = pose.reshape(pose.shape[:-1] + (-1, 2))

    if len(pose.shape) == 3:
        dir_vec = np.zeros((pose.shape[0], len(dir_vec_pairs_media), 2))
        for i, pair in enumerate(dir_vec_pairs_media):
            dir_vec[:, i] = pose[:, pair[1]] - pose[:, pair[0]]
            dir_vec[:, i, :] = normalize(dir_vec[:, i, :], axis=1)  # to unit length
    elif len(pose.shape) == 4:  # (batch, seq, ...)
        dir_vec = np.zeros((pose.shape[0], pose.shape[1], len(dir_vec_pairs_media), 2))
        for i, pair in enumerate(dir_vec_pairs_media):
            dir_vec[:, :, i] = pose[:, :, pair[1]] - pose[:, :, pair[0]]
        for j in range(dir_vec.shape[0]):  # batch
            for i in range(len(dir_vec_pairs)):
                dir_vec[j, :, i, :] = normalize(dir_vec[j, :, i, :], axis=1)  # to unit length
    else:
        assert False

    return dir_vec

def convert_pose_seq_to_dir_vec_mediapipe_hands_2d(pose):
    if pose.shape[-1] != 3:
        pose = pose.reshape(pose.shape[:-1] + (-1, 3))

    if len(pose.shape) == 3:
        dir_vec = np.zeros((pose.shape[0], len(dir_vec_pairs_media_hands), 2))
        for i, pair in enumerate(dir_vec_pairs_media_hands):
            dir_vec[:, i] = pose[:, pair[1]] - pose[:, pair[0]]
            dir_vec[:, i, :] = normalize(dir_vec[:, i, :], axis=1)  # to unit length
    elif len(pose.shape) == 4:  # (batch, seq, ...)
        dir_vec = np.zeros((pose.shape[0], pose.shape[1], len(dir_vec_pairs_media_hands), 2))
        for i, pair in enumerate(dir_vec_pairs_media_hands):
            dir_vec[:, :, i] = pose[:, :, pair[1]] - pose[:, :, pair[0]]
        for j in range(dir_vec.shape[0]):  # batch
            for i in range(len(dir_vec_pairs)):
                dir_vec[j, :, i, :] = normalize(dir_vec[j, :, i, :], axis=1)  # to unit length
    else:
        assert False

    return dir_vec