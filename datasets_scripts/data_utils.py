# ------------------------------------------------------------------------------
# Copyright (c) ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------

import glob
import matplotlib
import cv2
import re
import json
import _pickle as pickle
from webvtt import WebVTT
import numpy as np

COLOR = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}

## with detailed fingers
pairs_complex = [
    (0, 1, 'r'),
    (1, 2, 'r'),
    (1, 3, 'r'),
    (2, 5, 'r'),
    (5, 7, 'r'),
    (7, 9, 'r'),
    (9, 14, 'r'),
    (14, 15, 'r'),
    (15, 16, 'r'),
    (9, 17, 'r'),
    (17, 18, 'r'),
    (18, 19, 'r'),
    (9, 20, 'r'),
    (20, 21, 'r'),
    (21, 22, 'r'),
    (9, 23, 'r'),
    (23, 24, 'r'),
    (24, 25, 'r'),
    (9, 26, 'r'),
    (26, 27, 'r'),
    (27, 28, 'r'),
    (3, 6, 'r'),
    (6, 8, 'r'),
    (8, 10, 'r'),
    (10, 29, 'r'),
    (29, 30, 'r'),
    (30, 31, 'r'),
    (10, 32, 'r'),
    (32, 33, 'r'),
    (33, 34, 'r'),
    (10, 35, 'r'),
    (35, 36, 'r'),
    (36, 37, 'r'),
    (10, 38, 'r'),
    (38, 39, 'r'),
    (39, 40, 'r'),
    (10, 41, 'r'),
    (41, 42, 'r'),
    (42, 43, 'r'),
    (1, 44, 'r'),
    (44, 45, 'r'),
    (45, 47, 'r'),
    (44, 46, 'r'),
    (46, 48, 'r'),
]

## with simple fingers
pairs_simple = [
    (0, 1, 'r'),
    (1, 2, 'r'),
    (1, 3, 'r'),
    (2, 4, 'r'),
    (4, 6, 'r'),
    (6, 13, 'r'),
    (6, 14, 'r'),
    (6, 15, 'r'),
    (6, 16, 'r'),
    (6, 17, 'r'),

    (3, 5, 'r'),
    (5, 7, 'r'),
    (7, 18, 'r'),
    (7, 19, 'r'),
    (7, 20, 'r'),
    (7, 21, 'r'),
    (7, 22, 'r'),

    (1, 8, 'r'),
    (8, 9, 'r'),
    (8, 10, 'r'),
    (9, 11, 'r'),
    (10, 12, 'r'),
]

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

###############################################################################
def draw_3d_on_image(img, proj_joints, height, width, thickness=15):
    if proj_joints == []:
        return img

    new_img = img.copy()
    for pair in pairs_simple:
        pt1 = (int(proj_joints[pair[0]][0]), int(proj_joints[pair[0]][1]))
        pt2 = (int(proj_joints[pair[1]][0]), int(proj_joints[pair[1]][1]))
        if pt1[0] >= width or pt1[0] <= 0 or pt1[1] >= height or pt1[1] <= 0 or pt2[0] >= width or pt2[0] <= 0 or pt2[1] >= height or pt2[1] <= 0:
            pass
        else:
            rgb = [v * 255 for v in matplotlib.colors.to_rgba(pair[2])][:3]
            cv2.line(new_img, pt1, pt2, color=rgb[::-1], thickness=thickness)

    return new_img

# SKELETON
def draw_skeleton_on_image(img, skeleton, thickness=15):
    if not skeleton:
        return img

    new_img = img.copy()
    for pair in SkeletonWrapper.skeleton_line_pairs:
        pt1 = (int(skeleton[pair[0] * 3]), int(skeleton[pair[0] * 3 + 1]))
        pt2 = (int(skeleton[pair[1] * 3]), int(skeleton[pair[1] * 3 + 1]))
        if pt1[0] == 0 or pt2[1] == 0:
            pass
        else:
            rgb = [v * 255 for v in matplotlib.colors.to_rgba(pair[2])][:3]
            cv2.line(new_img, pt1, pt2, color=rgb[::-1], thickness=thickness)

    return new_img


def is_list_empty(my_list):
    return all(map(is_list_empty, my_list)) if isinstance(my_list, list) else False


def get_closest_skeleton(frame, selected_body):
    """ find the closest one to the selected skeleton """
    diff_idx = [i * 3 for i in range(8)] + [i * 3 + 1 for i in range(8)]  # upper-body

    min_diff = 10000000
    tracked_person = None
    for person in frame:  # people
        body = get_skeleton_from_frame(person)

        diff = 0
        n_diff = 0
        for i in diff_idx:
            if body[i] > 0 and selected_body[i] > 0:
                diff += abs(body[i] - selected_body[i])
                n_diff += 1
        if n_diff > 0:
            diff /= n_diff
        if diff < min_diff:
            min_diff = diff
            tracked_person = person

    base_distance = max(abs(selected_body[0 * 3 + 1] - selected_body[1 * 3 + 1]) * 3,
                        abs(selected_body[2 * 3] - selected_body[5 * 3]) * 2)
    if tracked_person and min_diff > base_distance:  # tracking failed
        tracked_person = None

    return tracked_person


def get_skeleton_from_frame(frame):
    # print(frame)
    if 'pose_keypoints_2d' in frame:
        # load keypoints of body, left hand, right hand
        openposes_real = frame["pose_keypoints_2d"]
        hands_left = frame["hand_left_keypoints_2d"]
        hands_right = frame["hand_right_keypoints_2d"]
            
        openpose_keypoints = []
        # delete miss matched finger points
        misses = [0,1*3,5*3,9*3,13*3,17*3]
        
        for i in range(0,len(openposes_real),3):
            tmp_keypoints = [openposes_real[i],openposes_real[i+1]]
            openpose_keypoints.append(tmp_keypoints)
            
        for i in range(0,len(hands_left),3):
            if i not in misses:
                tmp_keypoints = [hands_left[i],hands_left[i+1]]
                openpose_keypoints.append(tmp_keypoints)
            
        for i in range(0,len(hands_right),3):
            if i not in misses:
                tmp_keypoints = [hands_right[i],hands_right[i+1]]
                openpose_keypoints.append(tmp_keypoints)
                
        proj_joints = np.array(openpose_keypoints)  
        # only perserves upper body
        output = np.vstack(((proj_joints[0]+proj_joints[8])/2, proj_joints[1:3], proj_joints[5],proj_joints[3],proj_joints[6],proj_joints[4],proj_joints[7],proj_joints[40:55], proj_joints[25:40], proj_joints[0]))
        return output
    else:
        return None

class SkeletonWrapper:
    # color names: https://matplotlib.org/mpl_examples/color/named_colors.png
    visualization_line_pairs = [(0, 1, 'b'), (1, 2, 'darkred'), (2, 3, 'r'), (3, 4, 'gold'), (1, 5, 'darkgreen'), (5, 6, 'g'),
                                (6, 7, 'lightgreen'),
                                (1, 8, 'darkcyan'), (8, 9, 'c'), (9, 10, 'skyblue'), (1, 11, 'deeppink'), (11, 12, 'hotpink'), (12, 13, 'lightpink')]
    skeletons = []
    skeleton_line_pairs = [(0, 1, 'b'), (1, 2, 'darkred'), (2, 3, 'r'), (3, 4, 'gold'), (1, 5, 'darkgreen'),
                           (5, 6, 'g'), (6, 7, 'lightgreen')]

    def __init__(self, basepath, vid):
        # load skeleton data (and save it to pickle for next load)
        pickle_file = glob.glob(basepath + '/' + vid + '.pickle')

        if pickle_file:
            with open(pickle_file[0], 'rb') as file:
                self.skeletons = pickle.load(file)
        else:
            files = glob.glob(basepath + '/' + vid + '/*.json')
            if len(files) > 10:
                files = sorted(files)
                self.skeletons = []
                for file in files:
                    self.skeletons.append(self.read_skeleton_json(file))
                with open(basepath + '/' + vid + '.pickle', 'wb') as file:
                    pickle.dump(self.skeletons, file)
            else:
                self.skeletons = []


    def read_skeleton_json(self, file):
        with open(file) as json_file:
            skeleton_json = json.load(json_file)
            return skeleton_json['people']


    def get(self, start_frame_no, end_frame_no, interval=1):

        chunk = self.skeletons[start_frame_no:end_frame_no]

        if is_list_empty(chunk):
            return []
        else: 
            if interval > 1:
                return chunk[::int(interval)]
            else:
                return chunk


###############################################################################
# VIDEO
def read_video(base_path, vid):
    files = glob.glob(base_path + '/*' + vid + '.mp4')
    if len(files) == 0:
        return None
    elif len(files) >= 2:
        assert False
    filepath = files[0]

    video_obj = VideoWrapper(filepath)

    return video_obj

# VIDEO
def read_video_simple(vid):

    video_obj = VideoWrapper(vid)

    return video_obj


class VideoWrapper:
    video = []

    def __init__(self, filepath):
        self.filepath = filepath
        self.video = cv2.VideoCapture(filepath)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.framerate = self.video.get(cv2.CAP_PROP_FPS)

    def get_video_reader(self):
        return self.video

    def frame2second(self, frame_no):
        return frame_no / self.framerate

    def second2frame(self, second):
        return int(round(second * self.framerate))

    def set_current_frame(self, cur_frame_no):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, cur_frame_no)
