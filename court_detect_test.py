from utils import (read_video, 
                   read_fps,
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_meters_to_pixels_distance,
                   convert_pixels_distance_to_meters
                   )  

import constants
from trackers import PlayerTracker,BallTracker,PlayerTracker2
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import pandas as pd
from copy import deepcopy


def main():
    # Read video
    input_video = "MP4반포_실외.mp4"
    input_video_path = f'input_videos/{input_video}'
    video_frames,fps = read_video(input_video_path)

    #Court Line Detection model
    court_model_path = 'models/keypoints_model2.pth'
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    print(court_keypoints)
    #Draw Court Key Points
    output_video_frames = court_line_detector.draw_keypoints_on_video(video_frames, court_keypoints)
    

    output_video = input_video.replace(".mp4","")

    save_video(output_video_frames, f'output_videos/{output_video}_output2.mp4',fps)    
    #Test 입니다.
if __name__ == "__main__":
    main()