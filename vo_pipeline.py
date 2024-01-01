import cv2
import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
# from data_loader import DatasetLoader
from data_loader import KittiLoader, MalagaLoader, ParkingLoader, OwnDataLoader
from frame_state import FrameState, KeyPoint
from feature_extractor import FeatureExtractor
from visualizer import Visualizer
from estimate_campose import CamPoseEstimator

class VO_Pipeline:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.init_extractor = FeatureExtractor(extractor_type="sift")
        self.continuous_extractor = FeatureExtractor(extractor_type="shi-tomasi")
        self.visualizer = Visualizer()

        # get camera matrix
        self.K = self.dataloader.getCamera()
        self.pose_estimator = CamPoseEstimator(self.K)

        self.state = None
    
    def vo_initilization(self, frame_id_1, frame_id_2,):
        # get the first two frames
        image_1 = self.dataloader.getFrame(frame_id_1)
        image_2 = self.dataloader.getFrame(frame_id_2)

        # extract features
        kp1, des1 = self.init_extractor.extract(image_1, descibe=True)
        kp2, des2 = self.init_extractor.extract(image_2, descibe=True)
        matches = self.init_extractor.match(des1, des2)

        # get unmatched points in the second frame for candidate points
        mask_m = np.zeros(len(kp2), dtype=bool)
        mask_m[[m.trainIdx for m in matches]] = True
        kp2_um = kp2[~mask_m]

        # get the matched points (N, 2)
        kp1 = np.array([kp1[m.queryIdx] for m in matches])
        kp2 = np.array([kp2[m.trainIdx] for m in matches])

        # estimate camera pose
        M1 = np.eye(3, 4)
        M2, inlier_mask = self.pose_estimator.estimatePose(kp1, kp2)
        kp1 = kp1[inlier_mask.ravel()==1]
        kp2 = kp2[inlier_mask.ravel()==1]

        # triangulate the points (N, 3), (N, 2), (N, 2)
        points3d, kp1, kp2 = self.pose_estimator.triangulatePoints(kp1, kp2, M1, M2)

        # Visualize the 3D points and tracks
        self.visualizer.viewTracksCandidates(image_1, image_2, kp1, kp2, kp2_um)
        cam_poses = [M1, M2]
        self.visualizer.view3DPoints(points3d, cam_poses)

        # initialize the state
        self.state = FrameState()
        self.state.pose_history.append(M1)
        self.state.pose_history.append(M2)
        self.state.landmarks = points3d
        self.state.triangulated_kp = kp2
        self.state.candidate_kp = kp2_um
        self.state.candidate_kp_first = deepcopy(kp2_um)
        self.state.kp_first_pose = []
        for i in range(len(kp2_um)):
            self.state.kp_first_pose.append(M2)
        self.state.kp_first_pose = np.array(self.state.kp_first_pose)
        self.state.kp_track_length = np.ones(len(kp2_um))
        self.state.image = image_2

        self.state.landmark_history.append(len(points3d))
        self.state.landmarks_um = np.zeros((0, 3))

        print(f"Initialized VO pipeline.")
        print(self.state)
        

    def processFrame(self, image):

        # extend tracks for landmark keypoints from previous frame
        kp1 = self.state.triangulated_kp   # (N, 2)
        kp2, inlier_mask = self.continuous_extractor.track(
            image1=self.state.image,
            image2=image,
            points1=kp1)
        
        # remove unmatched landmarks, keypoints
        landmarks = self.state.landmarks[inlier_mask]
        landmarks_um = self.state.landmarks[~inlier_mask]
        kp1 = kp1[inlier_mask]
        kp2 = kp2[inlier_mask]

        # estimate camera pose of the current frame
        M1 = self.state.pose_history[-1]    # (3, 4)
        M2, inlier_mask = self.pose_estimator.estimatePosePnP(landmarks, kp2)

        # remove landmarks, keypoints that are outliers
        landmarks_um = np.append(landmarks_um, landmarks[~inlier_mask], axis=0)
        kp1 = kp1[inlier_mask]
        kp2 = kp2[inlier_mask]
        landmarks = landmarks[inlier_mask]

        # add new pose to the pose history
        self.state.pose_history.append(M2)

        # extend tracks for candidate keypoints from previous frame
        candidate_kp1 = self.state.candidate_kp
        candidate_kp2, inlier_mask = self.continuous_extractor.track(
            image1=self.state.image,
            image2=image,
            points1=candidate_kp1)
        
        # remove unmatched candidate keypoints, first pose, track length
        candidate_kp1 = candidate_kp1[inlier_mask]
        candidate_kp2 = candidate_kp2[inlier_mask]
        candidate_kp_first = self.state.candidate_kp_first[inlier_mask]
        first_pose = self.state.kp_first_pose[inlier_mask]
        track_length = self.state.kp_track_length[inlier_mask] + 1

        # triangulate new landmarks from candidate keypoints       
        extended_tracks = self.pose_estimator.triangulateCandidatePoints(
            candidate_kp=candidate_kp2,
            candidate_kp_first=candidate_kp_first,
            kp_first_pose=first_pose,
            kp_track_length=track_length,
            M2=M2)
        
        # add new landmarks to the state
        self.state.landmarks = np.append(landmarks, extended_tracks["landmarks"], axis=0)
        self.state.triangulated_kp = np.append(kp2, extended_tracks["landmarks_kp"], axis=0)
        self.state.candidate_kp = extended_tracks["candidate_kp"]
        self.state.candidate_kp_first = extended_tracks["candidate_kp_first"]
        self.state.kp_first_pose = extended_tracks["kp_first_pose"]
        self.state.kp_track_length = extended_tracks["kp_track_length"]
        self.state.image = image

        # Extract new features to add to the candidate keypoints
        current_keypoints = np.append(self.state.triangulated_kp, self.state.candidate_kp, axis=0)
        new_kp = self.continuous_extractor.extract(image, curr_kp=current_keypoints)
        new_first_pose = []
        for i in range(len(new_kp)):
            new_first_pose.append(M2)
        new_first_pose = np.array(new_first_pose)
        new_track_length = np.ones(len(new_kp))

        # add new keypoints to the state
        self.state.candidate_kp = np.append(self.state.candidate_kp, new_kp, axis=0)
        self.state.candidate_kp_first = np.append(self.state.candidate_kp_first, new_kp, axis=0)
        self.state.kp_first_pose = np.append(self.state.kp_first_pose, new_first_pose, axis=0)
        self.state.kp_track_length = np.append(self.state.kp_track_length, new_track_length)

        # update the landmark history
        self.state.landmark_history.append(len(self.state.landmarks))
        self.state.landmarks_um = np.append(self.state.landmarks_um, landmarks_um, axis=0)

        # Visualize the state
        self.visualizer.viewVOPipeline(self.state)


    def run(self):
        total_frames = self.dataloader.length
        init_frame_1 = 0
        init_frame_2 = 2 #3

        # Initialize the pipeline
        self.vo_initilization(init_frame_1, init_frame_2)

        # Process the remaining frames
        for frame_id in tqdm(range(init_frame_2+1, total_frames)):
        # for frame_id in range(init_frame_2+1, total_frames):
            image = self.dataloader.getFrame(frame_id)
            self.processFrame(image)


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.realpath(__file__))

    dataset_dir = os.path.join(cur_dir, "data")
    dataset_name = "parking"
    sequence_name = "05"

    # Load the datasets 
    if dataset_name == "kitti": dataloader = KittiLoader(dataset_dir)
    elif dataset_name == "malaga": dataloader = MalagaLoader(dataset_dir)
    elif dataset_name == "parking": dataloader = ParkingLoader(dataset_dir)
    elif dataset_name == "own": dataloader = OwnDataLoader(dataset_dir)
    
    # Load the dataset
    # dataloader = DatasetLoader(dataset_dir, dataset_name, sequence_name)
    vo_pipeline = VO_Pipeline(dataloader)
    vo_pipeline.run()