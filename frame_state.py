import numpy as np
import cv2
from collections import deque

class FrameState:
    def __init__(self, params):

        self.pose_history = []      # list of tracked poses
        self.landmarks = []         # triangulated 3d points  (N, 3)
        self.triangulated_kp = []   # corresponding keypoints (N, 2)
        self.candidate_kp = []      # candidate keypoints     (N, 2)
        self.candidate_kp_first = []# candidate keypoints in the first frame (N, 2)
        self.kp_first_pose = []     # poses of the first frame where the keypoint was detected (N, 3, 4)
        self.kp_track_length = []   # number of frames the keypoint has been tracked (N,)
        self.image = None           # image of the frame

        # for the visualizer
        self.landmark_history = []  # list of number of landmarks in each frame
        self.landmarks_um = []      # unmatched landmarks
        self.last_ba_call = 0 # Number of refined landmarks in the last ba call

        # FOR BUNDLE ADJUSTMENT
        self.ba_length = params["ba_frame_length"] # Number of frames to bundle adjust
        self.history = {}
        self.history["camera_poses"] = deque(maxlen=self.ba_length) # History of camera poses for the past 10 frames
        self.history["landmarks"] = deque(maxlen=self.ba_length) # History of landmarks for the past 10 frames
    
    def __str__(self) -> str:
        string = f"FrameState:\n"
        string += f"pose_history: {len(self.pose_history)}\n"
        string += f"landmarks: {self.landmarks.shape}\n"
        string += f"triangulated_kp: {self.triangulated_kp.shape}\n"
        string += f"candidate_kp: {self.candidate_kp.shape}\n"
        string += f"kp_first_pose: {self.kp_first_pose.shape}\n"
        string += f"kp_track_length: {self.kp_track_length.shape}\n"
        return string

class Landmark:
    def __init__(self, point_3d):
        self.point = point_3d
        self.keypoints = deque(maxlen=10) # list of keypoints of type 'KeyPoint' that correspond to this landmark
    def add_points(self, point_2d, cam_index):
        point_2d = KeyPoint(point_2d, cam_index) 
        self.keypoints.append(point_2d) 

class KeyPoint:
    def __init__(self, coord, cam_index):
        self.coord = coord
        self.cam_index = cam_index