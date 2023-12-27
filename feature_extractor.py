import numpy as np
import cv2
from copy import copy, deepcopy

class FeatureExtractor:
    def __init__(self):
        self.extractor_type = "sift" # ["sift", "surf", "orb", "fast", "harris"]
        self.lk_params = dict(winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))
    
    def extractHarrisCorners(self, image):
        """Extract Harris corners from the image with subpixel accuracy
        """

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = copy(image)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
        return corners
    
    def extractShiTomasiCorners(self, image):
        """Extract Shi-Tomasi corners from the image with subpixel accuracy
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = copy(image)
        gray = np.float32(gray)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.03, minDistance=7, blockSize=31)
        corners = corners.reshape(corners.shape[0], corners.shape[2])
        return corners
    
    def extractSiftFeatures(self, image, descibe=False):
        """Extract SIFT features from the image
        """
        sift = cv2.SIFT_create(nfeatures=1000)
        keypoints, descriptors = sift.detectAndCompute(image, None)
        keypoints = cv2.KeyPoint_convert(keypoints)

        if descibe:
            return keypoints, descriptors
        else:
            return keypoints
        
    def matchSiftFeatures(self, descriptors1, descriptors2):
        """Match SIFT features from the image
        """
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)
        return good
    
    def klt_tracker(self, image1, image2, points1):
        """Track the points using KLT tracker with bidirectional error check
        """
        points1 = np.array(points1, dtype=np.float32)

        points2, status, err = cv2.calcOpticalFlowPyrLK(image1, image2, points1, None, **self.lk_params)
        points1r, status, err = cv2.calcOpticalFlowPyrLK(image2, image1, points2, None, **self.lk_params)

        d = abs(points1 - points1r).reshape(-1, 2).max(-1)
        good = d < 30

        points1 = points1[good]
        points2 = points2[good]

        return points1, points2
    
    def klt_tracker_masked(self, image1, image2, points1):
        """Track the points using KLT tracker with bidirectional error check
        """
        points1 = np.array(points1, dtype=np.float32)

        points2, status, err = cv2.calcOpticalFlowPyrLK(image1, image2, points1, None, **self.lk_params)
        points1r, status, err = cv2.calcOpticalFlowPyrLK(image2, image1, points2, None, **self.lk_params)

        d = abs(points1 - points1r).reshape(-1, 2).max(-1)
        good = d < 30

        # points1 = points1[good]
        # points2 = points2[good]

        return points2, good