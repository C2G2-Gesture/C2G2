import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
import os
import json
from tqdm import tqdm
import glob
import pickle
from threading import Thread
from multiprocessing import Process
import subprocess
import shutil
# # For static images:
# IMAGE_FILES = []
# BG_COLOR = (192, 192, 192) # gray
# with mp_holistic.Holistic(
#     static_image_mode=True,
#     model_complexity=2,
#     enable_segmentation=True,
#     refine_face_landmarks=True) as holistic:
#   for idx, file in enumerate(IMAGE_FILES):
#     image = cv2.imread(file)
#     image_height, image_width, _ = image.shape
#     # Convert the BGR image to RGB before processing.
#     results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     if results.pose_landmarks:
#       print(
#           f'Nose coordinates: ('
#           f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
#           f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
#       )

#     annotated_image = image.copy()
#     # Draw segmentation on the image.
#     # To improve segmentation around boundaries, consider applying a joint
#     # bilateral filter to "results.segmentation_mask" with "image".
#     condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
#     bg_image = np.zeros(image.shape, dtype=np.uint8)
#     bg_image[:] = BG_COLOR
#     annotated_image = np.where(condition, annotated_image, bg_image)
#     # Draw pose, left and right hands, and face landmarks on the image.
#     mp_drawing.draw_landmarks(
#         annotated_image,
#         results.face_landmarks,
#         mp_holistic.FACEMESH_TESSELATION,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=mp_drawing_styles
#         .get_default_face_mesh_tesselation_style())
#     mp_drawing.draw_landmarks(
#         annotated_image,
#         results.pose_landmarks,
#         mp_holistic.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.
#         get_default_pose_landmarks_style())
#     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
#     # Plot pose world landmarks.
#     mp_drawing.plot_landmarks(
#         results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)

# # For webcam input:
# cap = cv2.VideoCapture(0)
# with mp_holistic.Holistic(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as holistic:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       break

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = holistic.process(image)

#     # Draw landmark annotation on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     mp_drawing.draw_landmarks(
#         image,
#         results.face_landmarks,
#         mp_holistic.FACEMESH_CONTOURS,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=mp_drawing_styles
#         .get_default_face_mesh_contours_style())
#     mp_drawing.draw_landmarks(
#         image,
#         results.pose_landmarks,
#         mp_holistic.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles
#         .get_default_pose_landmarks_style())
#     # mp_drawing.draw_landmarks(
#     #     image,
#     #     results.pose_landmarks,
#     #     mp_holistic.POSE_CONNECTIONS,
#     #     landmark_drawing_spec=mp_drawing_styles
#     #     .get_default_pose_landmarks_style())
#     # mp_drawing.draw_landmarks(
#     #     image,
#     #     results.pose_landmarks,
#     #     mp_holistic.POSE_CONNECTIONS,
#     #     landmark_drawing_spec=mp_drawing_styles
#     #     .get_default_pose_landmarks_style())
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()
def read_skeleton_json(_file):
    with open(_file) as json_file:
        skeleton_json = json.load(json_file)
        return skeleton_json

def save_skeleton_to_pickle(_vid,output_path):
    files = glob.glob(output_path + '/' + _vid + '/keypoints/*.json')
    if len(files) > 10:
        files = sorted(files)
        skeletons = []
        for file in files:
            skeletons.append(read_skeleton_json(file))
        with open(output_path + '/' + _vid + '/' + _vid + 'keypoints.pickle', 'wb') as file:
            filename = output_path + '/' + _vid + '/' + _vid + 'keypoints.pickle'
            print(f'save to {filename}')
            pickle.dump(skeletons, file)

def keypoints_extraction(image_path,debug=True):
    whole_path = os.path.split(image_path)[0]
    media_path = os.path.join(whole_path,"mediapipe_keypoints")
    media_visual_path = os.path.join(whole_path,"mediapipe_visual")
    
    if os.path.exists(media_path):
        shutil.rmtree(media_path)
    os.makedirs(media_path)
    
    if os.path.exists(media_visual_path):
        shutil.rmtree(media_visual_path)
        
    os.makedirs(media_visual_path)
    

    # For static images:
    IMAGE_FILES = []
    images = os.listdir(image_path)
    for image in images:
        if image.endswith(".png"):
            IMAGE_FILES.append(os.path.join(image_path,image))
            
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.3) as holistic:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            image_name = os.path.split(file)[-1]
            height, width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            save_json = os.path.join(media_path,image_name[:-4]+".json")
            save_dict = {}

            upper_poses_index = [11,12,13,14,15,16,23,24]
            upper_poses = []
            
            left_hands = []
            right_hands = []

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks!=None:
                for index in upper_poses_index:
                    if results.pose_landmarks.landmark[index].visibility>=0.05:
                        upper_poses.append(min(results.pose_landmarks.landmark[index].x*width,width))
                        upper_poses.append(min(results.pose_landmarks.landmark[index].y*height,height))
                        upper_poses.append(min(results.pose_landmarks.landmark[index].z*width,width))
                    else:
                        upper_poses.extend([0,0,0])

            if results.left_hand_landmarks!=None:
                for i in range(21):
                    # print(results.left_hand_landmarks.landmark[i])
                    if True:
                        left_hands.append(min(results.left_hand_landmarks.landmark[i].x*width,width))
                        left_hands.append(min(results.left_hand_landmarks.landmark[i].y*height,height))
                        left_hands.append(min(results.left_hand_landmarks.landmark[i].z*width,width))
            
            if results.right_hand_landmarks!=None:
                for i in range(21):
                    if True:
                        right_hands.append(min(results.right_hand_landmarks.landmark[i].x*width,width))
                        right_hands.append(min(results.right_hand_landmarks.landmark[i].y*height,height))
                        right_hands.append(min(results.right_hand_landmarks.landmark[i].z*width,width))  

            # print(upper_poses,len(upper_poses))
            
            save_dict["keypoints_3d_poses"] = upper_poses
            save_dict["keypoints_3d_left_hands"] = left_hands
            save_dict["keypoints_3d_right_hands"] = right_hands
            # use for debug
            if debug:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style())
                mp_drawing.draw_landmarks(
                    image,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style())
                mp_drawing.draw_landmarks(
                    image,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style())

                cv2.imwrite(f'{media_visual_path}/{image_name[:-4]}.png', image)

            with open(save_json, 'w') as f:
                f.write(json.dumps(save_dict))
            f.close()
            


def process_videos(mp4_videos,output_path):
    print(f"start thread to process for {len(mp4_videos)} videos")
    for i in iter(tqdm(mp4_videos)):
        keypoints_extraction(i,output_path)
        name = os.path.split(i)[-1][:-4]+'_'
        save_skeleton_to_pickle(name,output_path)


# all_videos = "/home/tiger/ted_expressive/videos_ted"
# all_videos = "/home/tiger/nfs/workspace/DiffGesture/test"
# mp4_videos = []
filter_json = "/home/tiger/nfs/workspace/yolov7/filtered.json"

output_path = "/home/tiger/nfs/workspace/render_dataset"

# with open(filter_json,"r") as f:
#     dict = f.read()
#     filtered_dict = json.loads(dict)
# passed_videos = filtered_dict["passed"]
# passed_videos = sorted(passed_videos)
# for filename in passed_videos:
#     if filename.endswith("mp4"):
#        whole_path = os.path.join(all_videos,filename) 
#        mp4_videos.append(whole_path)
# all_videos = "/home/tiger/nfs/ted_expressive/videos_ted"
# output_path = "/home/tiger/nfs/ted_expressive/media_process_v2"
# al_files = os.listdir(all_videos)
# mp4_videos = []
# for i in range(len(al_files)):
#     if al_files[i].endswith(".mp4"):
#         mp4_videos.append(os.path.join(all_videos,al_files[i]))
# # mp4_videos = passed_videos
# mp4_videos = mp4_videos
# num_threads = int(len(mp4_videos)/100)+1

# processes = []
# for i in range(num_threads):
#     if i*100>=len(mp4_videos):
#         tmp_videos=mp4_videos[100*i:]
#     else:
#         tmp_videos=mp4_videos[100*i:100*(i+1)]

#     processes.append(Process(target=process_videos,args=(tmp_videos,output_path,)))
    
# [p.start() for p in processes]

keypoints_extraction("/home/tiger/nfs/workspace/DiffGesture/openpose_dataset_v4/2BVSEnJte84__3433__3691/frames_cropped/2BVSEnJte84_3604_3691/frames")



