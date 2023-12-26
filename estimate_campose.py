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

        p_C2 = M2 @ np.vstack((points3d, np.ones((1, points3d.shape[1]))))
        mask = p_C2[2, :] > 0
        points3d = points3d[:, mask]

        return points3d.T