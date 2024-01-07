import cv2
import numpy as np
from glob import glob
import os
from matplotlib import pyplot as plt


# Load the images from the directory
# image_dir = "kitti05/kitti/05/image_0"
# image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
# images = [cv2.imread(file) for file in image_files]


class KLT_Tracker:

    def __init__(self):
        
        # Parameters for Harris Corner Detection
        
        self.blockSize = 2
        self.ksize = 3
        self.k = 0.04
        self.threshold = 0.1 # Increase this to get fewer distinguishable corners

        # Parameters for Lucas Kanade Tracker

        self.maxCorners = 200
        self.qualityLevel = 0.01
        self.minDistance = 7
        self.blockSize = 7
        self.winSize = (15, 15)
        self.maxLevel = 2
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01)
        self.feature_params = dict( maxCorners = self.maxCorners,
                                    qualityLevel = self.qualityLevel,
                                    minDistance = self.minDistance,
                                    blockSize = self.blockSize)
        self.lk_params = dict(winSize=self.winSize,
                              maxLevel=self.maxLevel,
                              criteria=self.criteria)

    def harris_corner_det(self, image):

        # Check if image is already grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Find Harris corners
        dst = cv2.cornerHarris(gray, self.blockSize, self.ksize, self.k)
        dst = cv2.dilate(dst, None)
        threshold = self.threshold * dst.max()
        dst = np.uint8(dst > threshold) * 255

        for j in range(0, dst.shape[0]):
            for i in range(0, dst.shape[1]):
                if(dst[j,i] > 0):
                    # image, center pt, radius, color, thickness
                    image_changed = cv2.circle(image, (i, j), 1, (0,255,0), 1)

        # Plot the image with the corners using plt
        plt.figure()
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(image_changed)
        plt.show()

        # return points
    
    def lucas_kanade_tracker(self, image1, image2):

        image_changed_1 = image1.copy()
        image_changed_2 = image2.copy()
        # Check if image is already grayscale
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = image1

        if len(image2.shape) == 3:
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = image2

        points1 = cv2.goodFeaturesToTrack(gray1, mask=None, useHarrisDetector=True, **self.feature_params)

        # Find the points in the second image
        points2, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, points1, None, **self.lk_params)

        # Select good points
        good_new = points2[status == 1]
        good_old = points1[status == 1]

        # Draw the tracks
        for _, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a, b, c, d = int(a), int(b), int(c), int(d)

            image1 = cv2.circle(image1, (a, b), 5, (0, 255, 0), 1)
            image_changed_2 = cv2.line(image_changed_1, (a, b), (c, d), (255, 0, 0), 2)
            image_changed_2 = cv2.circle(image_changed_2, (a, b), 5, (0, 255, 0), 1)

        # img = cv2.add(image_changed_1, image_changed_2)

        # Plot the image with the corners using plt
        plt.figure()
        plt.subplot(121)
        plt.imshow(image1)
        plt.subplot(122)
        plt.imshow(image_changed_2)
        plt.show()
        
        print(good_new.shape)

        return good_new
 
if __name__ == "__main__":
    
    tracker = KLT_Tracker()
    images = [cv2.imread("data/kitti/05/image_0/000000.png"), cv2.imread("data/kitti/05/image_0/000001.png")]
    # tracker.harris_corner_det(images[0])
    new_points = tracker.lucas_kanade_tracker(images[0], images[1])
    
