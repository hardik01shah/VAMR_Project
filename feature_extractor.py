import numpy as np
import cv2
from copy import copy, deepcopy

class FeatureExtractor:
    def __init__(self, extractor_type="sift", params=None):
        
        self.params = params
        self.lk_params = dict(winSize=(self.params["klt"]["winSize"], self.params["klt"]["winSize"]), 
                              maxLevel=self.params["klt"]["maxLevel"], 
                              criteria=(eval(self.params["klt"]["criteria"]["type"]),
                                        self.params["klt"]["criteria"]["maxCount"], 
                                        self.params["klt"]["criteria"]["epsilon"])
                            )

        self.extractor_type = extractor_type # ["sift", "surf", "orb", "fast", "harris"]
        self.extract = None
        if self.extractor_type == "sift":
            self.extract = self.extractSiftFeatures
            self.match = self.matchSiftFeatures

        elif self.extractor_type == "harris":
            self.extract = self.extractHarrisCorners
            self.match = None

        elif self.extractor_type == "shi-tomasi":
            self.extract = self.extractShiTomasiCorners
            self.match = None

        self.track = self.klt_tracker_masked
    
    def extractHarrisCorners(self, image, curr_kp=[], mask_radius=7):
        """Extract Harris corners from the image with subpixel accuracy
        """

        winSize = self.params["harris"]["winSize"]
        zeroZone = self.params["harris"]["zeroZone"]
        criteria_cc = self.params["harris"]["criteria"]
        blockSize = self.params["harris"]["blockSize"]
        ksize = self.params["harris"]["ksize"]
        k = self.params["harris"]["k"]
        thresh = self.params["harris"]["thresh"]

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = copy(image)

        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, blockSize, ksize, k)
        dst = cv2.dilate(dst, None)
        _, dst = cv2.threshold(dst, thresh * dst.max(), 255, 0)
        dst = np.uint8(dst)
        _, _, _, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (eval(criteria_cc["type"]), criteria_cc["maxCount"], criteria_cc["epsilon"])
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (winSize,winSize), (zeroZone,zeroZone), criteria)   

        # mask out the current keypoints to avoid overlapping features
        mask = np.ones(gray.shape, dtype=np.uint8) * 255
        for i in range(len(curr_kp)):
            cv2.circle(mask, (int(curr_kp[i][0]), int(curr_kp[i][1])), mask_radius, 0, -1)

        corners_int = corners.astype(np.int32) - 1
        corners = corners[mask[corners_int[:, 1], corners_int[:, 0]] == 255]

        print(corners.shape)   

        return corners
    
    def extractShiTomasiCorners(self, image, curr_kp=[], mask_radius=7):
        """Extract Shi-Tomasi corners from the image with subpixel accuracy
        """
        
        maxCorners = self.params["shi-tomasi"]["maxCorners"]
        qualityLevel = self.params["shi-tomasi"]["qualityLevel"]
        minDistance = self.params["shi-tomasi"]["minDistance"]
        blockSize = self.params["shi-tomasi"]["blockSize"]
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = copy(image)

        # mask out the current keypoints to avoid overlapping features
        mask = np.ones(gray.shape, dtype=np.uint8) * 255
        for i in range(len(curr_kp)):
            cv2.circle(mask, (int(curr_kp[i][0]), int(curr_kp[i][1])), mask_radius, 0, -1)

        gray = np.float32(gray)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance, mask=mask, blockSize=blockSize)
        corners = corners.reshape(corners.shape[0], corners.shape[2])
        return corners
    
    def extractSiftFeatures(self, image, descibe=False):
        """Extract SIFT features from the image
        """
        nfeatures = self.params["sift"]["nfeatures"]
        
        sift = cv2.SIFT_create(nfeatures)
        keypoints, descriptors = sift.detectAndCompute(image, None)
        keypoints = cv2.KeyPoint_convert(keypoints)

        if descibe:
            return keypoints, descriptors
        else:
            return keypoints
        
    def matchSiftFeatures(self, descriptors1, descriptors2):
        """Match SIFT features from the image
        """
        
        k = self.params["sift"]["k"]
        dist_thresh = self.params["sift"]["dist_threshold"]
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k)
        good = []
        for m, n in matches:
            if m.distance < dist_thresh * n.distance:
                good.append(m)
        return good
    
    def klt_tracker(self, image1, image2, points1, max_bidrectional_error=30):
        """Track the points using KLT tracker with bidirectional error check
        """
        dist_thresh = self.params["klt"]["dist_threshold"]
        # max_bidrectional_error = self.params["klt"]["max_bidrectional_error"]

        points1 = np.array(points1, dtype=np.float32)

        points2, status, err = cv2.calcOpticalFlowPyrLK(image1, image2, points1, None, **self.lk_params)
        points1r, status, err = cv2.calcOpticalFlowPyrLK(image2, image1, points2, None, **self.lk_params)

        d = abs(points1 - points1r).reshape(-1, 2).max(-1)
        good = d < dist_thresh

        points1 = points1[good]
        points2 = points2[good]

        return points1, points2
    
    def klt_tracker_masked(self, image1, image2, points1, max_bidrectional_error=30):
        """Track the points using KLT tracker with bidirectional error check
        """
        
        dist_thresh = np.inf
        # max_bidrectional_error = self.params["klt"]["max_bidrectional_error"]

        points1 = np.array(points1, dtype=np.float32)

        points2, status, err = cv2.calcOpticalFlowPyrLK(image1, image2, points1, None, **self.lk_params)
        points1r, status, err = cv2.calcOpticalFlowPyrLK(image2, image1, points2, None, **self.lk_params)

        d = abs(points1 - points1r).reshape(-1, 2).max(-1)
        good = d < dist_thresh

        assert len(points1) == len(points2) == len(good)

        return points2, good