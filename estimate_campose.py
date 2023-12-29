import numpy as np
import cv2

class CamPoseEstimator:
    def __init__(self, camera_matrix):
        self.K = camera_matrix

    def estimatePose(self, points1, points2):
        """Estimate the pose from 2D-2D correspondences using RANSAC.
        """
        
        assert points1.shape == points2.shape
        assert len(points1.shape) == 2 and points1.shape[1] == 2 # points1.shape = (N, 2)

        # Estimate the essential matrix
        E, mask = cv2.findEssentialMat(points1, points2, self.K, method=cv2.RANSAC, prob=0.9999, threshold=1.0)

        inlier_points1 = points1[mask.ravel()==1]
        inlier_points2 = points2[mask.ravel()==1]

        # Recover the pose from the essential matrix
        _, R, t, _ = cv2.recoverPose(E, inlier_points1, inlier_points2, self.K)
        M = np.hstack((R, t))

        return M, mask
    
    def getAngleBetweenBearingVectors(self, p3d, M1, M2):
        """Calculate the angle between two bearing vectors joining the camera center and the 3D point.
        """
        assert p3d.shape == (3,)
        assert M1.shape == M2.shape == (3, 4)

        # get the bearing vectors - p3d projected onto the image plane of the two cameras
        p_C1 = M1 @ np.hstack((p3d, 1))
        p_C2 = M2 @ np.hstack((p3d, 1))

        # get the baseline vector (b) - vector joining the two camera centers
        M_21 = np.vstack((M2, np.array([0,0,0,1])) )@ np.linalg.inv(np.vstack((M1, np.array([0,0,0,1]))))
        assert np.allclose(M_21[3, :], np.array([0,0,0,1]))
        b = np.linalg.norm(M_21[:3, 3]/M_21[3, 3])

        # get the angle between the two bearing vectors - cosine rule (a^2 = b^2 + c^2 - 2bc*cos(angle))
        p1 = np.linalg.norm(p_C1)
        p2 = np.linalg.norm(p_C2)

        angle = np.rad2deg(np.arccos((p1**2 + p2**2 - b**2) / (2*p1*p2)))

        return angle
    
    def triangulatePoints(self, points1, points2, M1, M2):
        assert points1.shape == points2.shape
        assert len(points1.shape) == 2 and points1.shape[1] == 2 # points1.shape = (N, 2)

        # triangulatePoints() requires the points to be in the shape of (2, N) and type of np.float32
        points1 = (points1.T).astype(np.float32)
        points2 = (points2.T).astype(np.float32)

        # Construct the projection matrices
        P1 = self.K @ M1
        P2 = self.K @ M2

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

        # return the points in the shape of (N, 3) and (N, 2)
        return points3d.T, points1.T, points2.T
    
    def triangulatePointsMasked(self, points1, points2, M1, M2):
        assert points1.shape == points2.shape
        assert len(points1.shape) == 2 and points1.shape[1] == 2 # points1.shape = (N, 2)

        # triangulatePoints() requires the points to be in the shape of (2, N) and type of np.float32
        points1 = (points1.T).astype(np.float32)
        points2 = (points2.T).astype(np.float32)

        # Construct the projection matrices
        P1 = self.K @ M1
        P2 = self.K @ M2

        # Triangulate the points
        points4d = cv2.triangulatePoints(P1, P2, points1, points2)
        points3d = points4d[:3, :] / points4d[3, :]

        # remove points that are behind the camera
        p_C1 = M1 @ np.vstack((points3d, np.ones((1, points3d.shape[1]))))
        mask_1 = p_C1[2, :] > 0
        
        p_C2 = M2 @ np.vstack((points3d, np.ones((1, points3d.shape[1]))))
        mask_2 = p_C2[2, :] > 0

        mask = np.logical_and(mask_1, mask_2)

        # return the points in the shape of (N, 3) and (N, 2)
        return points3d.T, mask
    
    def triangulateCandidatePoints(self, candidate_kp, candidate_kp_first, kp_first_pose, kp_track_length, M2):
        """Triangulate the candidate keypoints of the current frame.
        Args:
            candidate_kp: (N, 2) coordinates of the candidate keypoints in the current frame
            candidate_kp_first: (N, 2) coordinates of the candidate keypoints in the first frame
            kp_first_pose: (N, 3, 4) poses of the first frame where the keypoint was detected
            kp_track_length: (N,) number of frames the keypoint has been tracked
            M2: (3, 4) pose of the current frame
        """
        assert candidate_kp.shape == candidate_kp_first.shape
        assert len(candidate_kp) == len(candidate_kp_first) == len(kp_first_pose) == len(kp_track_length)

        # divide the candidate keypoints into two groups based on the number of frames they have been tracked
        triangulate_mask = kp_track_length >= 3 # 2

        kp_tmp = candidate_kp[triangulate_mask]             # Try triangulating
        kp_first_tmp = candidate_kp_first[triangulate_mask]
        kp_poses = kp_first_pose[triangulate_mask]
        kp_lens = kp_track_length[triangulate_mask]

        candidate_kp_tmp = candidate_kp[~triangulate_mask]          # Keep for next frame
        candidate_kp_first_tmp = candidate_kp_first[~triangulate_mask]
        kp_first_pose_tmp = kp_first_pose[~triangulate_mask]
        kp_track_length_tmp = kp_track_length[~triangulate_mask]

        # new landmarks
        landmarks = np.zeros((0, 3))
        landmarks_kp = np.zeros((0, 2))

        # triangulate based on first frame pose
        ff_poses, indx = np.unique(kp_poses, axis=0, return_index=True)

        for i in range(len(ff_poses)):
            M1 = ff_poses[i]
            m1_indices = np.where(np.all(kp_poses == M1, axis=(1,2)))[0]

            m1_kps = kp_tmp[m1_indices]            # (N, 2)
            m1_kp_firsts = kp_first_tmp[m1_indices]
            m1_lens = kp_lens[m1_indices]

            # try triangulating all these points first
            points3d, mask = self.triangulatePointsMasked(m1_kp_firsts, m1_kps, M1, M2)

            # remove points that are behind the camera
            points3d = points3d[mask]
            m1_kps = m1_kps[mask]
            m1_kp_firsts = m1_kp_firsts[mask]
            m1_lens = m1_lens[mask]

            # remove points that have angle between bearing vectors > threshold
            for j in range(len(points3d)):
                p3d = points3d[j]
                angle = self.getAngleBetweenBearingVectors(p3d, M1, M2)
                print(angle)
                if angle > 3:  # 4
                    landmarks = np.append(landmarks, p3d.reshape(1,-1), axis=0)
                    landmarks_kp = np.append(landmarks_kp, m1_kps[j].reshape(1,-1), axis=0)
                else:
                    candidate_kp_tmp = np.append(candidate_kp_tmp, m1_kps[j].reshape(1,-1), axis=0)
                    candidate_kp_first_tmp = np.append(candidate_kp_first_tmp, m1_kp_firsts[j].reshape(1,-1), axis=0)
                    kp_first_pose_tmp = np.append(kp_first_pose_tmp, M1.reshape(1,3,4), axis=0)
                    kp_track_length_tmp = np.append(kp_track_length_tmp, m1_lens[j])

        extended_tracks = {
            "landmarks": landmarks,
            "landmarks_kp": landmarks_kp,
            "candidate_kp": candidate_kp_tmp,
            "candidate_kp_first": candidate_kp_first_tmp,
            "kp_first_pose": kp_first_pose_tmp,
            "kp_track_length": kp_track_length_tmp
        }
        return extended_tracks


    
    def estimatePosePnP(self, points3d, points2d):
        assert len(points3d.shape) == 2 and points3d.shape[1] == 3
        assert len(points2d.shape) == 2 and points2d.shape[1] == 2

        # solvePnPRansac requires the points to be in the shape of (N, 2) and (N, 3) and type of np.float32
        # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e
        points3d = (points3d).astype(np.float32)
        points2d = (points2d).astype(np.float32)

        # Estimate the pose using PnP
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints = points3d,
            imagePoints = points2d,
            cameraMatrix = self.K,
            distCoeffs=None,
            reprojectionError=5.0,
            iterationsCount=1000000,
            confidence=0.9999)
        R = cv2.Rodrigues(rvec)[0]
        M = np.hstack((R, tvec))

        # get inlier mask
        inliers = inliers.ravel()
        good_mask = np.zeros(len(points2d), dtype=bool)
        good_mask[inliers] = True

        return M, good_mask