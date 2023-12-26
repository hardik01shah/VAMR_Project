import os
import glob
import numpy as np
import cv2

class DatasetLoader:
    def __init__(self, dataset_dir, dataset_name, sequence_name=None):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.sequence_name = sequence_name
        
        if self.dataset_name == "kitti":
            if self.sequence_name is None:
                self.sequence_name = "05"

            self.image_dir = os.path.join(self.dataset_dir, self.dataset_name, self.sequence_name, "image_0")
            self.cam_calib_file = os.path.join(self.dataset_dir, self.dataset_name, self.sequence_name, "calib.txt")


        self.length = len(glob.glob(os.path.join(self.image_dir, "*.png")))

    def getFrame(self, frame_id, grayscale=True):
        image_file = os.path.join(self.image_dir, "%06d.png" % frame_id)
        if grayscale:
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_file)
        return image
    
    def getCamera(self):
        if self.dataset_name == "kitti":
            with open(self.cam_calib_file, "r") as f:
                lines = f.readlines()
                P0 = lines[0].strip().split(" ")[1:]
                P0 = np.array(P0, dtype=np.float32)
                P0 = P0.reshape((3, 4))
                P0 = P0[:3, :3]
                return P0