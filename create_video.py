import cv2
import numpy as np
import os
import glob

class VideoProcessor:

    def __init__(self, outpath):
        """
        Initialize the video processor.

        :param input_video: The path to the input video file.
        :param output_video: The path to the output video file.
        """
        self.out_vid = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'MP4V'), 25, (1200, 600))

    def process_video(self, image_dir):

        # Process a video file frame by frame and detect faces using Amazon Rekognition.
        # Write the output video file with bounding boxes around the faces.

        image_list = glob.glob(os.path.join(image_dir, "*.png"))
        num_images = len(image_list)

        for count in range(num_images):
            image_name = image_dir + "/vo_pipeline_state_{}.png".format(count)
            frame = cv2.imread(image_name)
            self.out_vid.write(frame)
            print("Processed frame {}".format(count))

        self.out_vid.release()

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    image_dir = cur_dir + "/malaga"
    output_video_path = cur_dir + "/malaga.mp4"
    video_processor = VideoProcessor(output_video_path)
    video_processor.process_video(image_dir)