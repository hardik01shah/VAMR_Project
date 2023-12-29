import numpy as np
import cv2

class FrameState:
    def __init__(self):
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
    
    def __str__(self) -> str:
        string = f"FrameState:\n"
        string += f"pose_history: {len(self.pose_history)}\n"
        string += f"landmarks: {self.landmarks.shape}\n"
        string += f"triangulated_kp: {self.triangulated_kp.shape}\n"
        string += f"candidate_kp: {self.candidate_kp.shape}\n"
        string += f"kp_first_pose: {self.kp_first_pose.shape}\n"
        string += f"kp_track_length: {self.kp_track_length.shape}\n"
        return string

class KeyPoint:
    def __init__(self, coord, first_pose, track_history):
        self.coord = coord                         # (u, v) coordinates of the keypoint (2,1)
        self.first_pose = first_pose               # pose of the first frame where the keypoint was detected
        self.track_history = np.array(coord)       # uv coordinates of tracked keypoints in previous frames (2, N)
        