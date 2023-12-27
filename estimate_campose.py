import numpy as np
import cv2

class CamPoseEstimator:
    def __init__(self):
        pass

    def estimatePose(self, points1, points2, camera_matrix):
        
        assert points1.shape == points2.shape
        assert len(points1.shape) == 2 and points1.shape[1] == 2 # points1.shape = (N, 2)

        # Estimate the essential matrix
        E, mask = cv2.findEssentialMat(points1, points2, camera_matrix, method=cv2.RANSAC, prob=0.9999, threshold=1.0)

        inlier_points1 = points1[mask.ravel()==1]
        inlier_points2 = points2[mask.ravel()==1]

        # Recover the pose from the essential matrix
        _, R, t, _ = cv2.recoverPose(E, inlier_points1, inlier_points2, camera_matrix)

        return R, t, mask
    
    def triangulatePoints(self, points1, points2, camera_matrix, M1, M2):
        assert points1.shape == points2.shape
        assert len(points1.shape) == 2 and points1.shape[1] == 2 # points1.shape = (N, 2)

        # triangulatePoints() requires the points to be in the shape of (2, N) and type of np.float32
        points1 = (points1.T).astype(np.float32)
        points2 = (points2.T).astype(np.float32)

        # Construct the projection matrices
        P1 = camera_matrix @ M1
        P2 = camera_matrix @ M2

        # Triangulate the points
        points4d = cv2.triangulatePoints(P1, P2, points1, points2)
        points3d = points4d[:3, :] / points4d[3, :]

        # remove points that are behind the camera
        p_C1 = M1 @ np.vstack((points3d, np.ones((1, points3d.shape[1]))))
        mask = p_C1[2, :] > 0
        points3d = points3d[:, mask]
        points1 = points1[:, mask]
        points2 = points2[:, mask]

        p_C2 = M2 @ np.vstack((points3d, np.ones((1, points3d.shape[1]))))
        mask = p_C2[2, :] > 0
        points3d = points3d[:, mask]
        points1 = points1[:, mask]
        points2 = points2[:, mask]

        return points3d.T, points1.T, points2.T
    
    def estimatePosePnP(self, points3d, points2d, camera_matrix):
        assert len(points3d.shape) == 2 and points3d.shape[1] == 3
        assert len(points2d.shape) == 2 and points2d.shape[1] == 2

        # solvePnPRansac requires the points to be in the shape of (N, 2) and (N, 3) and type of np.float32
        # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e
        points3d = (points3d).astype(np.float32)
        points2d = (points2d).astype(np.float32)

        print(points3d.shape)
        print(points2d.shape)

        # Estimate the pose using PnP
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints = points3d,
            imagePoints = points2d,
            cameraMatrix = camera_matrix,
            distCoeffs=None,
            reprojectionError=5.0,
            iterationsCount=1000000,
            confidence=0.9999)
        R = cv2.Rodrigues(rvec)[0]
        return R, tvec, inliers