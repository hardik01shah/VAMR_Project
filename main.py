import numpy as np
import cv2
import os

from data_loader import DatasetLoader
from klt_tracker import KLT_Tracker
from visualizer import Visualizer
from feature_extractor import FeatureExtractor
from estimate_campose import CamPoseEstimator


if __name__ == "__main__":

    cur_dir = os.path.dirname(os.path.realpath(__file__))

    dataset_dir = os.path.join(cur_dir, "data")
    dataset_name = "kitti"
    sequence_name = "05"
    frame_id = 0

    # Load the dataset
    dataset_loader = DatasetLoader(dataset_dir, dataset_name, sequence_name)
    image = dataset_loader.getFrame(frame_id)
    K = dataset_loader.getCamera()
    print(f"Loaded camera matrix:\n{K}")
    
    # Visualize the points
    visualizer = Visualizer()
    visualizer.viewImage(image)

    # Extract features
    feature_extractor = FeatureExtractor()
    # points = feature_extractor.extractHarrisCorners(image)
    points = feature_extractor.extractShiTomasiCorners(image)
    # points = feature_extractor.extractSiftFeatures(image)

    # Visualize the points
    visualizer.viewPoints(image, points)


    image_1 = dataset_loader.getFrame(0)
    image_2 = dataset_loader.getFrame(1)

    # Track the points
    points = feature_extractor.extractShiTomasiCorners(image)
    points1, points2 = feature_extractor.klt_tracker(image_1, image_2, points)

    # Visualize the tracks
    visualizer.viewTracks(image_1, image_2, points1, points2)
    

    # Match features
    keypoints_1, descriptors_1 = feature_extractor.extractSiftFeatures(image_1, descibe=True)
    keypoints_2, descriptors_2 = feature_extractor.extractSiftFeatures(image_2, descibe=True)
    matches = feature_extractor.matchSiftFeatures(descriptors_1, descriptors_2)

    # Visualize the matches
    visualizer.viewMatches(image_1, image_2, keypoints_1, keypoints_2, matches)

    # get the matched points
    points1 = np.array([keypoints_1[m.queryIdx] for m in matches])
    points2 = np.array([keypoints_2[m.trainIdx] for m in matches])

    # Estimate the camera pose
    cam_pose_estimator = CamPoseEstimator()
    R, t, mask = cam_pose_estimator.estimatePose(points1, points2, K)

    # calculate inlier points
    inlier_points1 = points1[mask.ravel()==1]
    inlier_points2 = points2[mask.ravel()==1]

    # Visualize the inlier matches
    visualizer.viewTracks(image_1, image_2, inlier_points1, inlier_points2)

    # Triangulate the points
    M1 = np.eye(3, 4)
    M2 = np.hstack((R, t))
    points3d, inlier_points1, inlier_points2 = cam_pose_estimator.triangulatePoints(inlier_points1, inlier_points2, K, M1, M2)

    # Visualize the 3D points
    cam_poses = [M1, M2]
    visualizer.view3DPoints(points3d, cam_poses)

    # Estimate the pose of third frame using PnP
    image_3 = dataset_loader.getFrame(2)

    # Extract features using KLTracker
    inlier_points3, good_mask = feature_extractor.klt_tracker_masked(image_2, image_3, inlier_points2)
    assert len(good_mask) == len(inlier_points2)
    inlier_points2 = inlier_points2[good_mask]
    inlier_points3 = inlier_points3[good_mask]

    R, t, inliers = cam_pose_estimator.estimatePosePnP(points3d, inlier_points3, K)
    points3d = points3d[inliers.ravel()]
    
    # Visualize the 3D points
    M3 = np.hstack((R, t))
    cam_poses = [M1, M2, M3]
    visualizer.view3DPoints(points3d, cam_poses)
    





