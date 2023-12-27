import numpy as np
import cv2
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self):
        pass

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
    
    def view3DPoints(self, points3d, cam_poses):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim3d(-100,100)
        ax.set_ylim3d(-100,100)
        ax.set_zlim3d(-100,100)
        
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