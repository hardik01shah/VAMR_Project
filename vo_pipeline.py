import cv2
import os
import numpy as np
from scipy.spatial.distance import pdist
from tqdm import tqdm
from copy import deepcopy
from collections import deque
import yaml
import argparse
# from data_loader import DatasetLoader
from data_loader import KittiLoader, MalagaLoader, ParkingLoader, OwnDataLoader
from frame_state import FrameState, KeyPoint, Landmark
from feature_extractor import FeatureExtractor
from visualizer import Visualizer
from estimate_campose import CamPoseEstimator
from bundle_adjust import BundleAdjuster
import itertools

class VO_Pipeline:
    def __init__(self, dataloader, config_file):

        # Extracting parameters for the pipelines
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.params = config
        feature_extractor_params = config["feature_extractor"]
        pose_estimator_params = config["pose_estimator"]

        self.min_track_length = config["min_track_length"]
        self.angle_threshold = config["angle_threshold"]
        self.mask_radius = config["mask_radius"]
        self.init_frame_1 = config["init_frame_1"]
        self.init_frame_2 = config["init_frame_2"]
        self.dataset_name = config["dataset_name"]
        self.sequence_name = config["sequence_name"]

        self.dataloader = dataloader
        init_extractor_type = feature_extractor_params["init_extractor_type"]
        cont_extractor_type = feature_extractor_params["cont_extractor_type"]
        self.init_extractor = FeatureExtractor(extractor_type=init_extractor_type, params=feature_extractor_params)
        self.continuous_extractor = FeatureExtractor(extractor_type=cont_extractor_type, params=feature_extractor_params)
        self.visualizer = Visualizer(self.params["bundle_adjustment"]["use_ba"])

        # get camera matrix
        self.K = self.dataloader.getCamera()
        self.pose_estimator = CamPoseEstimator(self.K, pose_estimator_params)

        # Initialize Bundle Adjuster
        self.use_ba = self.params["bundle_adjustment"]["use_ba"]
        self.ba_call_rate = self.params["bundle_adjustment"]["ba_call_rate"]
        self.BA = BundleAdjuster(self.K, self.params["bundle_adjustment"]["ba_params"])

        # Initialize bootstrap
        self.use_bootstrap = self.params["bootstrap"]["use_bootstrap"]
        self.bootstrap_index = self.params["bootstrap"]["bootstrap_index"]
        self.bootstrap_frequency = self.params["bootstrap"]["bootstrap_frequency"]
        self.bootstrap_lm_threshold = self.params["bootstrap"]["bootstrap_lm_threshold"]

        if self.use_ba and self.use_bootstrap:
            raise ValueError("Bundle adjustment and bootstrapping cannot be used together.")

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

        # estimate camera pose and create a mask for the inliers got from RANSAC
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
        self.state = FrameState(self.params["bundle_adjustment"])
        self.state.pose_history.append(M1)
        self.state.pose_history.append(M2)
        self.state.landmarks = points3d
        self.state.triangulated_kp = kp2

        # ---------------------------------------------------------------
        # FOR BUNDLE ADJUSTMENT
        # ---------------------------------------------------------------
        # Add landmarks, keypoints and the camera poses to the history for bundle adjustment
        if self.use_ba:       
            landmarks_list = [Landmark(points3d[i]) for i in range(len(points3d))]
            for i in range(len(landmarks_list)):
                landmarks_list[i].add_points(kp2[i], len(self.state.pose_history)-2)

            self.state.history["camera_poses"].append(M2)
            self.state.history["landmarks"].append(landmarks_list) # Appending the list with triangulated landmarks

        # ---------------------------------------------------------------
        # FOR BOOTSTRAPPING
        # ---------------------------------------------------------------
        # add landmarks to the history
        if self.use_bootstrap:
            cur_landmarks = [Landmark(points3d[i]) for i in range(len(points3d))]
            for i in range(len(kp2)):
                cur_landmarks[i].add_points(kp2[i], 1)
            self.state.history["landmarks"].append(cur_landmarks)

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
        self.state.last_ba_call = 0

        print(f"Initialized VO pipeline.")
        print(self.state)
        

    def processFrame(self, image):

        # extend tracks for landmark keypoints from previous frame
        # Gives a boolean inlier_mask
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
        
        # ---------------------------------------------------------------
        # FOR BUNDLE ADJUSTMENT
        # ---------------------------------------------------------------
        # Separating the landmarks that are inlier and outliers
        # Outliers are saved in the state history at first and inliers are added to the state later 
        # along with newly triangulated landmarks
        if self.use_ba:
            landmarks_ba_inlier = np.asarray(self.state.history["landmarks"][-1])[inlier_mask] 
            landmarks_ba_outlier = np.asarray(self.state.history["landmarks"][-1])[~inlier_mask]
            self.state.history["landmarks"][-1] = list(landmarks_ba_outlier) 
        # ---------------------------------------------------------------

        # estimate camera pose of the current frame
        M1 = self.state.pose_history[-1]    # (3, 4)
        M2, inlier_mask = self.pose_estimator.estimatePosePnP(landmarks, kp2)

        # remove landmarks, keypoints that are outliers
        landmarks_um = np.append(landmarks_um, landmarks[~inlier_mask], axis=0)
        kp1 = kp1[inlier_mask]
        kp2 = kp2[inlier_mask]
        landmarks = landmarks[inlier_mask]

        # ---------------------------------------------------------------
        # FOR BUNDLE ADJUSTMENT
        # ---------------------------------------------------------------
        # Separating the landmarks that are inlier and outliers again
        if self.use_ba:
            self.state.history["landmarks"][-1] += list(landmarks_ba_inlier[~inlier_mask])
            landmarks_ba_inlier = landmarks_ba_inlier[inlier_mask]
        # ---------------------------------------------------------------
                
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
            M2=M2,
            min_track_length=self.min_track_length,
            angle_threshold=self.angle_threshold)
        
        # add new landmarks to the state
        self.state.landmarks = np.append(landmarks, extended_tracks["landmarks"], axis=0)
        self.state.triangulated_kp = np.append(kp2, extended_tracks["landmarks_kp"], axis=0)
        self.state.candidate_kp = extended_tracks["candidate_kp"]
        self.state.candidate_kp_first = extended_tracks["candidate_kp_first"]
        self.state.kp_first_pose = extended_tracks["kp_first_pose"]
        self.state.kp_track_length = extended_tracks["kp_track_length"]
        self.state.image = image

        # ---------------------------------------------------------------
        # FOR BUNDLE ADJUSTMENT
        # ---------------------------------------------------------------
        # Adding tracked inlier landmarks and newly triangulated landmarks to the state history (along with keypoints)
        if self.use_ba:
            landmarks_list1 = []
            landmarks_list2 = []
            for i in range(len(landmarks_ba_inlier)):
                landmarks_ba_inlier[i].add_points(kp2[i], len(self.state.pose_history)-2)
                landmarks_list1.append(landmarks_ba_inlier[i])
            for i in range(len(extended_tracks["landmarks"])):
                landmarks_list2.append(Landmark(extended_tracks["landmarks"][i]))
            for i in range(len(landmarks_list2)):
                landmarks_list2[i].add_points(extended_tracks["landmarks_kp"][i], len(self.state.pose_history)-2)

            self.state.history["camera_poses"].append(self.state.pose_history[-1])
            self.state.history["landmarks"].append(landmarks_list1+landmarks_list2)
        # ---------------------------------------------------------------
        
        # ---------------------------------------------------------------
        # FOR BOOTSTRAPPING
        # ---------------------------------------------------------------
        if self.use_bootstrap:
            # add landmarks to the history
            cur_landmarks = [Landmark(self.state.landmarks[i]) for i in range(len(self.state.landmarks))]
            for i in range(len(self.state.triangulated_kp)):
                cur_landmarks[i].add_points(self.state.triangulated_kp[i], len(self.state.pose_history)-1)
            self.state.history["landmarks"].append(cur_landmarks)

        # Extract new features to add to the candidate keypoints
        current_keypoints = np.append(self.state.triangulated_kp, self.state.candidate_kp, axis=0)
        new_kp = self.continuous_extractor.extract(image, curr_kp=current_keypoints, mask_radius=self.mask_radius)
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

        # ---------------------------------------------------------------
        # FOR BUNDLE ADJUSTMENT
        # ---------------------------------------------------------------
        # Adjust camera poses and landmarks using bundle adjustment
        if len(self.state.pose_history) % self.ba_call_rate == 0 and self.use_ba:
            poses_refined, landmarks_refined = self.BA.bundle_adjust(self.state.history["camera_poses"], self.state.history["landmarks"])
            self.state.pose_history[-len(poses_refined):] = poses_refined
            self.state.landmarks_um = np.append(self.state.landmarks_um[:self.state.last_ba_call], landmarks_refined, axis=0)
            self.state.last_ba_call = len(self.state.landmarks_um)
        # ---------------------------------------------------------------


        # Visualize the state
        self.visualizer.viewVOPipeline(self.state, self.dataset_name)
    
    def bootstrap(self, prev_indx, prev_img, cur_img):
        prev_landmarks = self.state.history["landmarks"][prev_indx]
        cur_landmarks = self.state.history["landmarks"][-1]

        # get the landmarks that are common in both frames
        common_landmarks = np.zeros((0, 3))
        prev_uv = np.zeros((0, 2))
        cur_uv = np.zeros((0, 2))
        
        for i in range(len(cur_landmarks)):
            for j in range(len(prev_landmarks)):
                if np.array_equal(cur_landmarks[i].point, prev_landmarks[j].point):
                    common_landmarks = np.append(common_landmarks, cur_landmarks[i].point.reshape(1,3), axis=0)
                    prev_uv = np.append(prev_uv, prev_landmarks[j].keypoints[-1].coord.reshape(1,2), axis=0)
                    cur_uv = np.append(cur_uv, cur_landmarks[i].keypoints[-1].coord.reshape(1,2), axis=0)
                    
        # estimate camera pose by bootstrapping
        M1 = self.state.pose_history[prev_indx]
        M2, inlier_mask = self.pose_estimator.estimatePose(prev_uv, cur_uv)

        # remove unmatched landmarks, keypoints
        common_landmarks = common_landmarks[inlier_mask.ravel()==1]
        prev_uv = prev_uv[inlier_mask.ravel()==1]
        cur_uv = cur_uv[inlier_mask.ravel()==1]

        # triangulate the points (N, 3), (N, 2), (N, 2)
        points3d, mask = self.pose_estimator.triangulatePointsMasked(prev_uv, cur_uv, np.eye(3,4), M2)

        # Visualize the 3D points and tracks
        # self.visualizer.viewTracksCandidates(prev_img, cur_img, prev_uv, cur_uv, np.zeros((0,2)))
        # cam_poses = [np.eye(3,4), M2]
        # self.visualizer.view3DPoints(points3d, cam_poses)

        # ensure that the triangulated points are in front of the camera
        common_landmarks = common_landmarks[mask]
        points3d = points3d[mask]
        prev_uv = prev_uv[mask]
        cur_uv = cur_uv[mask]

        assert len(common_landmarks) == len(points3d) == len(prev_uv) == len(cur_uv)

        # calculate relative scale factor
        # calulate distance between pairs of points from 1) Current VO pipeline 2) Bootstrapped VO pipeline
        dist_vo = pdist(common_landmarks)
        dist_bootstrap = pdist(points3d)

        # remove zero distances
        mask_vo = dist_vo != 0
        mask_bootstrap = dist_bootstrap != 0
        mask = mask_vo & mask_bootstrap
        dist_vo = dist_vo[mask]
        dist_bootstrap = dist_bootstrap[mask]

        # calculate the ratio of the distances
        ratio = np.mean(dist_vo/dist_bootstrap)
        # ratio = np.median(dist_vo/dist_bootstrap)
        
        old_M2 = self.state.pose_history[-1]
        vo_t = np.linalg.norm(old_M2[:,3] - M1[:,3])
        bootstrap_t = np.linalg.norm(M2[:,3])    # should be close to 2

        assert np.allclose(bootstrap_t, 1)

        print(f"VO Distance: {vo_t}")
        print(f"Bootstrap Distance: {bootstrap_t}")
        print(f"Ratio: {ratio}")
        print(f"New Distance: {ratio * np.linalg.norm(M2[:,3])}")

        # estimate new pose by scaling the bootstrapped pose
        M2 = np.hstack((M2[:, :3], M2[:, 3].reshape(3,1)*ratio))

        # M2 is the relative pose of the current frame wrt the previous frame, so we need to get the absolute pose
        # make M1 and M2 homogeneous
        M1_4x4 = np.vstack((M1, np.array([0, 0, 0, 1])))    # T_Cam1_World
        M2_4x4 = np.vstack((M2, np.array([0, 0, 0, 1])))    # T_Cam2_Cam1
        M2 = (M2_4x4 @ M1_4x4)                              # T_Cam2_World
        M2 = M2[:3, :]/M2[3, 3]                             

        # update pose in the pose history
        self.state.pose_history[-1] = M2

        # get new landmarks and reset the candidate keypoints

        # extract features
        kp1, des1 = self.init_extractor.extract(prev_img, descibe=True)
        kp2, des2 = self.init_extractor.extract(cur_img, descibe=True)
        matches = self.init_extractor.match(des1, des2)

        # get unmatched points in the second frame for candidate points
        mask_m = np.zeros(len(kp2), dtype=bool)
        mask_m[[m.trainIdx for m in matches]] = True
        kp2_um = kp2[~mask_m]

        # get the matched points (N, 2)
        kp1 = np.array([kp1[m.queryIdx] for m in matches])
        kp2 = np.array([kp2[m.trainIdx] for m in matches])

        _, inlier_mask = self.pose_estimator.estimatePose(kp1, kp2)
        kp1 = kp1[inlier_mask.ravel()==1]
        kp2 = kp2[inlier_mask.ravel()==1]

        # triangulate the points (N, 3), (N, 2), (N, 2)
        points3d, kp1, kp2 = self.pose_estimator.triangulatePoints(kp1, kp2, M1, M2)

        # Visualize the 3D points and tracks
        # self.visualizer.viewTracksCandidates(prev_img, cur_img, kp1, kp2, kp2_um)
        # cam_poses = [M1, old_M2]
        # self.visualizer.view3DPoints(points3d, cam_poses)

        # update the state
        self.state.landmarks = points3d
        self.state.triangulated_kp = kp2
        self.state.candidate_kp = kp2_um
        self.state.candidate_kp_first = deepcopy(kp2_um)
        self.state.kp_first_pose = []
        for i in range(len(kp2_um)):
            self.state.kp_first_pose.append(M2)
        self.state.kp_first_pose = np.array(self.state.kp_first_pose)
        self.state.kp_track_length = np.ones(len(kp2_um))
        self.state.image = cur_img

        # add landmarks to the history
        cur_landmarks = [Landmark(points3d[i]) for i in range(len(points3d))]
        for i in range(len(kp2)):
            cur_landmarks[i].add_points(kp2[i], 1)
        self.state.history["landmarks"][-1] = cur_landmarks
    
    def bootstrap_landmarks(self, prev_indx, prev_img, cur_img):
        # get new landmarks and reset the candidate keypoints
        # extract features
        kp1, des1 = self.init_extractor.extract(prev_img, descibe=True)
        kp2, des2 = self.init_extractor.extract(cur_img, descibe=True)
        matches = self.init_extractor.match(des1, des2)

        # get unmatched points in the second frame for candidate points
        mask_m = np.zeros(len(kp2), dtype=bool)
        mask_m[[m.trainIdx for m in matches]] = True
        kp2_um = kp2[~mask_m]

        # get the matched points (N, 2)
        kp1 = np.array([kp1[m.queryIdx] for m in matches])
        kp2 = np.array([kp2[m.trainIdx] for m in matches])

        new_m2, inlier_mask = self.pose_estimator.estimatePose(kp1, kp2)
        kp1 = kp1[inlier_mask.ravel()==1]
        kp2 = kp2[inlier_mask.ravel()==1]

        # triangulate the points (N, 3), (N, 2), (N, 2)
        points3d, kp1, kp2 = self.pose_estimator.triangulatePoints(kp1, kp2, np.eye(3,4), new_m2)

        M1 = self.state.pose_history[prev_indx]
        old_M2 = self.state.pose_history[-1]
        M1_4x4 = np.vstack((M1, np.array([0, 0, 0, 1])))        # T_Cam1_World
        M2_4x4 = np.vstack((old_M2, np.array([0, 0, 0, 1])))    # T_Cam2_Cam1

        # convert points3d to world frame
        points3d = np.vstack((points3d.T, np.ones((1, len(points3d)))))
        points3d = np.linalg.inv(M1_4x4) @ points3d
        points3d = points3d[:3, :]/points3d[3, :]
        points3d = points3d.T

        # Visualize the 3D points and tracks
        # self.visualizer.viewTracksCandidates(prev_img, cur_img, kp1, kp2, kp2_um)
        # cam_poses = [M1, old_M2]
        # self.visualizer.view3DPoints(points3d, cam_poses)

        # update the state
        self.state.landmarks = points3d
        self.state.triangulated_kp = kp2
        self.state.candidate_kp = kp2_um
        self.state.candidate_kp_first = deepcopy(kp2_um)
        self.state.kp_first_pose = []
        for i in range(len(kp2_um)):
            self.state.kp_first_pose.append(old_M2)
        self.state.kp_first_pose = np.array(self.state.kp_first_pose)
        self.state.kp_track_length = np.ones(len(kp2_um))
        self.state.image = cur_img

        # add landmarks to the history
        cur_landmarks = [Landmark(points3d[i]) for i in range(len(points3d))]
        for i in range(len(kp2)):
            cur_landmarks[i].add_points(kp2[i], 1)
        self.state.history["landmarks"][-1] = cur_landmarks

    def run(self):
        total_frames = self.dataloader.length
        init_frame_1 = self.params["init_frame_1"]
        init_frame_2 = self.params["init_frame_2"]

        # Initialize the pipeline
        self.vo_initilization(init_frame_1, init_frame_2)

        # Process the remaining frames
        for frame_id in tqdm(range(init_frame_2+1, total_frames)):
            image = self.dataloader.getFrame(frame_id)
            self.processFrame(image)

            # ---------------------------------------------------------------
            # FOR BOOTSTRAPPING
            # ---------------------------------------------------------------
            if self.use_bootstrap and (frame_id - 2*init_frame_2) > self.bootstrap_index:
                if len(self.state.landmarks) < self.bootstrap_lm_threshold:
                    prev_img = self.dataloader.getFrame(frame_id - self.bootstrap_index + 1)
                    self.bootstrap(-self.bootstrap_index, prev_img, image)
                    # self.bootstrap_landmarks(self.bootstrap_index, prev_img, image)


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.realpath(__file__))

    # Create argparser
    parser = argparse.ArgumentParser(description="Visual Odometry Pipeline")
    parser.add_argument("--dataset_dir", type=str, default=os.path.join(cur_dir, "data"), help="Path to the dataset directory")
    parser.add_argument("--dataset_name", type=str, default="kitti", help="Name of the dataset: can be kitti, malaga, parking or own")
    parser.add_argument("--config", type=str, default="config/kitti.yaml", 
            help="Path to the config file: can be config/params.yaml, config/params_kitti.yaml, config/params_malaga.yaml, config/params_parking.yaml or config/params_own.yaml")
    
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name
    config_file = args.config

    # Load the datasets 
    if dataset_name == "kitti": dataloader = KittiLoader(dataset_dir)
    elif dataset_name == "malaga": dataloader = MalagaLoader(dataset_dir)
    elif dataset_name == "parking": dataloader = ParkingLoader(dataset_dir)
    elif dataset_name == "own": dataloader = OwnDataLoader(dataset_dir)
    
    # Load the dataset
    vo_pipeline = VO_Pipeline(dataloader, config_file)
    vo_pipeline.run()