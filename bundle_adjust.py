import numpy as np
import time
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import cv2
from visualizer import Visualizer

class BundleAdjuster:
    
    def __init__(self, K, params):
        
        self.K = K
        self.visualizer = Visualizer()
        self.x_scale = params["x_scale"]
        self.ftol = params["ftol"]
        self.method = params["method"]
        self.verbose = params["verbose"]

    def convert_to_bundle_adjust_format(self, pose_history, landmarks_history):
        """
        Get the 3D landmark objects and corresponding camera poses from the history
        
        Args:
            pose_history (list): A list of numpy arrays of size 3x4 each representing camera poses.
            landmarks_history (list): A list of Landmark objects representing landmarks from each frame.
        
        Returns:
            camera_params (numpy.ndarray): Camera parameters, shape (n_cameras, 9)
            points_3d (numpy.ndarray): 3D points, shape (n_points, 3)
            camera_ind (numpy.ndarray): Camera indices, shape (n_observations,)
            point_ind (numpy.ndarray): 3D point indices, shape (n_observations,)
            points_2d (numpy.ndarray): 2D points, shape (n_observations, 2)
        """
        
        n_cameras = len(pose_history)
        landmarks_list = [l for landmarks in list(landmarks_history) for l in landmarks]
        n_points = len(landmarks_list)

        cam_poses = np.array(list(pose_history))
        camera_params = np.zeros((n_cameras, 9))
        points_3d = np.zeros((n_points, 3))
        camera_ind = []
        point_ind = []
        points_2d = []

        print("n_cameras: ", n_cameras)
        print("n_landmarks: ", len(landmarks_list))

        for i, pose in enumerate(cam_poses):
            # Extract rotation and translation from pose
            rvec, _ = cv2.Rodrigues(pose[:, :3])
            tvec = pose[:, 3]

            # Set camera parameters
            camera_params[i, :3] = rvec.flatten()
            camera_params[i, 3:6] = tvec.flatten()
            camera_params[i, 6] = self.K[0, 0]  # focal distance
            camera_params[i, 7:] = [0, 0]  # distortion parameters

        for i, landmark in enumerate(landmarks_list):
            points_3d[i] = landmark.point
            # Set observation indices and 2D coordinates
            for keypoint in landmark.keypoints:
                camera_ind.append(keypoint.cam_index)
                point_ind.append(i)
                points_2d.append(keypoint.coord)

        if np.max(camera_ind) > n_cameras - 1:
            mask = camera_ind > np.max(camera_ind) - n_cameras
            camera_ind = np.array(camera_ind)[mask]
            camera_ind = camera_ind - np.min(camera_ind)
            point_ind = np.array(point_ind)[mask]
            points_2d = np.array(points_2d)[mask]
        else:
            camera_ind = np.array(camera_ind)
            point_ind = np.array(point_ind)
            points_2d = np.array(points_2d)

        return camera_params, cam_poses, points_3d, camera_ind, point_ind, points_2d
        
    def project(self, points_3d, camera_pose):
        """
        Project 3D points to 2D using camera parameters
        
        Args:
            points_3d (numpy.ndarray): 3D points to be projected, shape (N, 3)
            K (numpy.ndarray): Camera intrinsic matrix, shape (3, 3)
            camera_pose (numpy.ndarray): Camera pose matrix, shape (3, 4)
        
        Returns:
            numpy.ndarray: Projected 2D points, shape (2, N)
        """
        # ... implementation of the projection function ...
        points_proj = np.zeros((points_3d.shape[0], 2))
        for i in range(points_3d.shape[0]):
            T = np.vstack((camera_pose[i], np.array([0, 0, 0, 1])))
            points_cam = T @ np.append(points_3d[i], 1) # (4,)
            points_3d_cam = points_cam[:3] / points_cam[3] # (3,)
            temp = self.K @ points_3d_cam # (3,)
            points_proj[i] = temp[:2] / temp[2] # (2, 1)

        return points_proj # (2, N)
    
   
    def get_hom_trans(self, camera_params):
        """
        Convert camera parameters to a 3x4 homogeneous transformation matrix.
        Args:
            camera_params (numpy.ndarray): Camera parameters, shape (n_cameras, 9)
        Returns:
            numpy.ndarray: Homogeneous transformation matrix, shape (n_cameras, 3, 4)
        """
        n_cameras = camera_params.shape[0]
        homogeneous_transforms = np.zeros((n_cameras, 3, 4))

        for i in range(n_cameras):
            rvec = camera_params[i, :3]
            tvec = camera_params[i, 3:6]
            R, _ = cv2.Rodrigues(rvec)
            # Create homogeneous transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec
            homogeneous_transforms[i] = T[:3, :]

        return homogeneous_transforms

        
    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        """
        Calculates the residual error between the projected 3D points and the observed 2D points.

        Args:
            params (ndarray): Array of camera parameters and 3D points.
            n_cameras (int): Number of cameras.
            n_points (int): Number of 3D points.
            camera_indices (ndarray): Array of camera indices for each point.
            point_indices (ndarray): Array of point indices for each camera.
            points_2d (ndarray): Array of observed 2D points.

        Returns:
            ndarray: Residual error between the projected 3D points and the observed 2D points.
        """
        camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
        camera_poses = self.get_hom_trans(camera_params)
        points_3d = params[n_cameras * 9:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_poses[camera_indices])

        return (points_proj-points_2d).ravel()
    
    def bundle_adjustment_sparsity(self, n_cameras, n_points, camera_indices, point_indices):
        """
        Compute the sparsity pattern of the Jacobian matrix for bundle adjustment.

        Parameters:
        - n_cameras (int): Number of cameras.
        - n_points (int): Number of 3D points.
        - camera_indices (ndarray): Array of camera indices.
        - point_indices (ndarray): Array of point indices.

        Returns:
        - A (lil_matrix): Sparse matrix representing the sparsity pattern of the Jacobian matrix.
        """
        
        m = camera_indices.size * 2
        n = n_cameras * 9 + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(9):
            A[2 * i, camera_indices * 9 + s] = 1
            A[2 * i + 1, camera_indices * 9 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

        return A


    def bundle_adjust(self, pose_history, landmarks_history):
        """
        Perform bundle adjustment to optimize camera poses and 3D landmark positions.

        Args:
            pose_history (list): List of camera poses.
            landmarks_history (list): List of landmarks observed in each frame.

        Returns:
            Optimized camera poses (numpy.ndarray)
            Optimized 3D landmark positions (numpy.ndarray)
        """

        # Convert the data to the format required by the bundle adjustment function
        camera_params, _, points_3d, camera_ind, point_ind, points_2d = self.convert_to_bundle_adjust_format(pose_history, landmarks_history)

        n_cameras = len(pose_history)
        n_points = points_3d.shape[0]

        x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
        f0 = self.fun(x0, n_cameras, n_points, camera_ind, point_ind, points_2d)

        A = self.bundle_adjustment_sparsity(n_cameras, n_points, camera_ind, point_ind)

        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-2, method='trf',
                            args=(n_cameras, n_points, camera_ind, point_ind, points_2d))

        camera_params = res.x[:n_cameras * 9].reshape((n_cameras, 9))
        camera_poses = self.get_hom_trans(camera_params)
        points_3d = res.x[n_cameras * 9:].reshape((n_points, 3))

        return camera_poses, points_3d
