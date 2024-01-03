import numpy as np
from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
from frame_state import FrameState, KeyPoint

class Visualizer:
    def __init__(self):
        self.indx = 0

    def viewImage(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        plt.imshow(image)
        plt.show()
    
    def viewPoints(self, image, points):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        plt.imshow(image)
        plt.scatter(points[:,0], points[:,1], s=1, c='r', marker='x')
        plt.show()
    
    def viewMatches(self, image1, image2, points1, points2, matches):
        # image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
        # image = np.concatenate((image1, image2), axis=1)
        plt.imshow(image2)
        # plt.scatter(points1[:,0], points1[:,1], s=1, c='r', marker='x')
        plt.scatter(points2[:,0], points2[:,1], s=1, c='r', marker='x')
        for i in range(len(matches)):
            plt.plot([points1[matches[i].queryIdx,0], points2[matches[i].trainIdx,0]], 
                     [points1[matches[i].queryIdx,1], points2[matches[i].trainIdx,1]], 
                     'c-', linewidth=1.5)
        plt.show()
    
    def viewTracks(self, image1, image2, points1, points2):
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
        plt.imshow(image2)
        plt.scatter(points2[:,0], points2[:,1], s=1, c='r', marker='x')

        for i in range(len(points2)):
            plt.plot([points1[i,0], points2[i,0]], 
                     [points1[i,1], points2[i,1]], 
                     'c-', linewidth=1.5)
        plt.show()
    
    def viewTracksCandidates(self, image1, image2, points1, points2, points2_um):
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
        fig = plt.figure(figsize=(12,6))
        plt.title("Matched and Unmatched Keypoints for VO initialization")
        plt.imshow(image2)
        plt.scatter(points2[:,0], points2[:,1], s=10, c='y', marker='x', linewidths=0.5)
        plt.scatter(points2_um[:,0], points2_um[:,1], s=10, c='r', marker='x', linewidths=0.5)
        plt.legend(["Matched Keypoints", "Unmatched Keypoints"], loc="lower right", fontsize=8)

        for i in range(len(points2)):
            plt.plot([points1[i,0], points2[i,0]], 
                     [points1[i,1], points2[i,1]], 
                     'g--', linewidth=1.0)
        plt.show()
    
    def view3DPoints(self, points3d, cam_poses):
        
        fig = plt.figure(figsize=(12,6))
        plt.title("Camera pose and triangulated 3D points")
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim3d(-10,10)
        ax.set_ylim3d(-10,10)
        ax.set_zlim3d(-10,10)
        
        ax.scatter(points3d[:,0], points3d[:,1], points3d[:,2], s=1, c='r', marker='o')
        for i in range(len(cam_poses)):
            M = cam_poses[i]
            R = M[:,:3]
            t = M[:,3]
            self.PlotCamera(R, t, ax=ax)
        plt.show()
    
    def PlotCamera(self, R, t, ax=None, scale=1.0, color='b'):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        Rcw = R.transpose()
        tcw = -Rcw @ t

        # Define a path along the camera gridlines
        camera_points = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [1, -1, 1],
            [-1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
            [1, -1, 1],
            [0, 0, 0],
            [-1, -1, 1],
            [0, 0, 0],
            [-1, 1, 1]
        ])

        # Make sure that this vector has the right shape
        tcw = np.reshape(tcw, (3, 1))

        cam_points_world = (Rcw @ (scale * camera_points.transpose()) + np.tile(tcw, (1, 12))).transpose()

        ax.plot(xs=cam_points_world[:,0], ys=cam_points_world[:,1], zs=cam_points_world[:,2], color=color)

        plt.show(block=False)

        return ax
    
    def getPositionFromPose(self, trajectory: np.ndarray):
        """Get the position from the trajectory.
        Poses are transformations that convert points from the camera frame to the world frame.
        The position of the camera is the translation component of the inverse of this homography.
        Args:
            trajectory (np.ndarray): trajectory of the camera
        Returns:
            np.ndarray: position of the camera
        """
        postions = np.zeros((len(trajectory), 3))
        for i in range(len(trajectory)):
            M = trajectory[i]
            R = M[:,:3]
            t = M[:,3].reshape(3,1)

            postions[i] = (-R.T @ t).T
        
        return postions
    
    def viewVOPipeline(self, state: FrameState):
        """Visualize the VO pipeline state
        Args:
            state (FrameState): state of the VO pipeline
            landmarks_um (np.array): unmarked landmarks
        """

        # Visualize 4 subplots:
        #   1. Image overlayed with triangulated keypoints and candidate keypoints
        #   2. Global trajectory - 3D points(projected to 2D) and camera poses
        #   3. Number of landmarks in each frame
        #   4. Local trajectory

        fig = plt.figure(figsize=(12,6))
        fig.suptitle("VO Pipeline State", fontsize=16)

        # 1. Image overlayed with triangulated keypoints and candidate keypoints
        ax = fig.add_subplot(221)
        plt.title(f"Landmarks and Candidate Keypoints in Current Frame: {self.indx}")
        im = deepcopy(state.image)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax.imshow(im)

        # triangulated keypoints
        kp1 = state.triangulated_kp
        ax.scatter(kp1[:,0], kp1[:,1], s=2, c='g', marker='x', linewidths=0.5,
                   label="Triangulated Keypoints",
                   facecolor=None)

        # candidate keypoints
        kp2 = state.candidate_kp
        ax.scatter(kp2[:,0], kp2[:,1], s=2, c='r', marker='x', linewidths=0.5,
                   label="Candidate Keypoints",
                   facecolor=None)
        ax.set_xlim(0, im.shape[1])
        ax.set_ylim(im.shape[0], 0)
        ax.legend(loc="lower right", fontsize=8)

        # remove the ticks
        plt.xticks([])
        plt.yticks([])

        # 2. Global trajectory - 3D points(projected to 2D) and camera poses
        positions = self.getPositionFromPose(state.pose_history)
        ax = fig.add_subplot(122)
        plt.title("Full Trajectory and Landmarks")

        # plot the trajectory
        ax.scatter(positions[:,0], positions[:,2], s=8, c='b', marker='o', facecolor=None)
        ax.set_aspect('equal')
        ax.set_adjustable('datalim')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # plot the 3d points
        p3d = np.append(state.landmarks, state.landmarks_um, axis=0)
        ax.scatter(p3d[:,0], p3d[:,2], s=0.5, c='g', alpha=0.05, facecolor=None)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # 3. Number of landmarks in each frame
        ax = fig.add_subplot(245)
        plt.title("# Landmarks in each frame", fontsize=8)
        ax.plot(state.landmark_history)

        # 4. Local trajectory
        ax = fig.add_subplot(246)
        plt.title("Trajectory of Last 20 Frames", fontsize=8)
        local_positions = positions[-20:]
        ax.scatter(local_positions[:,0], local_positions[:,2], s=8, c='b', marker='x', facecolor=None)
        ax.set_aspect('equal')
        ax.set_adjustable('datalim')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # plot the 3d points
        p3d = state.landmarks
        ax.scatter(p3d[:,0], p3d[:,2], s=4, marker='o', c='g', facecolor=None)
        
        xlim2 = ax.get_xlim()
        ylim2 = ax.get_ylim()
        xlim = np.array(xlim)
        ylim = np.array(ylim)
        if xlim2[0] < xlim[0]:
            xlim[0] -= min(xlim[0] - xlim2[0], 10)
        if xlim2[1] > xlim[1]:
            xlim[1] += min(xlim2[1] - xlim[1], 10)
        if ylim2[0] < ylim[0]:
            ylim[0] -= min(ylim[0] - ylim2[0], 10)
        if ylim2[1] > ylim[1]:
            ylim[1] += min(ylim2[1] - ylim[1], 10)

        # increase xlim and ylim by 30%
        # xlim = np.array(xlim)
        # ylim = np.array(ylim)
        # xlen = xlim[1] - xlim[0]
        # ylen = ylim[1] - ylim[0]
        # xlim = xlim + np.array([-xlen*0.8, xlen*0.8])
        # ylim = ylim + np.array([-ylen*0.8, ylen*0.8])
        # xlim = tuple(xlim)
        # ylim = tuple(ylim)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # convert plot to image
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        cv2.imshow("VO Pipeline State", data)
        cv2.waitKey(1)
        plt.savefig("own/vo_pipeline_state_{}.png".format(self.indx))
        self.indx += 1
        plt.close(fig)
