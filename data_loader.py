import os
import glob
import numpy as np
import cv2

class KittiLoader:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.dataset_name = "kitti"
        self.sequence_name = "05"

        self.image_dir = os.path.join(self.dataset_dir, self.dataset_name, self.sequence_name, "image_0")
        self.cam_calib_file = os.path.join(self.dataset_dir, self.dataset_name, self.sequence_name, "calib.txt")

        # self.length = len(glob.glob(os.path.join(self.image_dir, "*.png")))
        self.image_list = glob.glob(os.path.join(self.image_dir, "*.png"))
        self.length = len(self.image_list)
        self.image_list.sort()

    def getFrame(self, frame_id, grayscale=True):
        image_file = self.image_list[frame_id]    
        print(image_file)    
        if grayscale:
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_file)
        return image
    
    def getCamera(self):
        with open(self.cam_calib_file, "r") as f:
            lines = f.readlines()
            P0 = lines[0].strip().split(" ")[1:]
            P0 = np.array(P0, dtype=np.float32)
            P0 = P0.reshape((3, 4))
            P0 = P0[:3, :3]
            return P0
            
class MalagaLoader:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.dataset_name = "malaga"
        self.sequence_name = None

        self.image_dir = os.path.join(self.dataset_dir, self.dataset_name, "malaga-urban-dataset-extract-07_rectified_800x600_Images")
        self.cam_calib_file = os.path.join(self.dataset_dir, self.dataset_name, "camera_params_rectified_a=0_800x600.txt")

        # self.length = len(glob.glob(os.path.join(self.image_dir, "*.png")))
        self.image_list = glob.glob(os.path.join(self.image_dir, "*left.jpg"))
        self.length = len(self.image_list)
        self.image_list.sort()


    def getFrame(self, frame_id, grayscale=True):
        image_file = self.image_list[frame_id]
        if grayscale:
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_file)
        return image
    
    def getCamera(self):
        with open(self.cam_calib_file, "r") as f:
            lines = f.readlines()
            cx = lines[6].strip().split("=")[1]
            cy = lines[7].strip().split("=")[1]
            fx = lines[8].strip().split("=")[1]
            fy = lines[9].strip().split("=")[1]
            P0 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            return P0
            
class ParkingLoader:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.dataset_name = "parking"
        self.sequence_name = None

        self.image_dir = os.path.join(self.dataset_dir, self.dataset_name, "images")
        self.cam_calib_file = os.path.join(self.dataset_dir, self.dataset_name, "K.txt")

        # self.length = len(glob.glob(os.path.join(self.image_dir, "*.png")))
        self.image_list = glob.glob(os.path.join(self.image_dir, "*.png"))
        self.length = len(self.image_list)
        self.image_list.sort()

    def getFrame(self, frame_id, grayscale=True):
        image_file = self.image_list[frame_id]
        if grayscale:
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_file)
        return image
    
    def getCamera(self):
        with open(self.cam_calib_file, "r") as f:
            P0 = np.array([[331.37, 0, 320], [0, 369.568, 240], [0, 0, 1]])
            return P0  

class OwnDataLoader:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.dataset_name = "own"
        self.sequence_name = None

        self.image_dir = os.path.join(self.dataset_dir, self.dataset_name, "images_left")
        self.cam_calib_file = os.path.join(self.dataset_dir, self.dataset_name, "calib.txt")

        # self.length = len(glob.glob(os.path.join(self.image_dir, "*.png")))
        self.image_list = glob.glob(os.path.join(self.image_dir, "*.jpg"))
        self.image_list.sort()
        self.length = len(self.image_list)
        print(self.image_list)

    def getFrame(self, frame_id, grayscale=True):
        image_file = self.image_list[frame_id]
        if grayscale:
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_file)
        return image
    
    def getCamera(self):
        with open(self.cam_calib_file, "r") as f:
            lines = f.readlines()
            fx = lines[0].strip().split("=")[1]
            fy = lines[1].strip().split("=")[1]
            cx = lines[2].strip().split("=")[1]
            cy = lines[3].strip().split("=")[1]
            P0 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            return P0  