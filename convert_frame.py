
import cv2
import numpy as np
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




def convert_fps(video_frames, target_fps, original_fps):
    new_frames = []
    frame_interval = original_fps / target_fps

    for i in range(len(video_frames) - 1):
        new_frames.append(video_frames[i])
        # 보간하여 새로운 프레임 생성
        alpha = 0.5  # 중간 프레임 생성
        interpolated_frame = cv2.addWeighted(video_frames[i], 1 - alpha, video_frames[i + 1], alpha, 0)
        new_frames.append(interpolated_frame)

    new_frames.append(video_frames[-1])  # 마지막 프레임 추가
    return new_frames


if __name__ == "__main__":
    # Read video
    input_video = "MP4반포_실외.mp4"
    input_video_path = f'input_videos/{input_video}'
    video_frames,fps = read_video(input_video_path)
    target_fps = 60
    video_frames_60fps = convert_fps(video_frames, target_fps, original_fps=fps)

    output_video = input_video.replace(".mp4","")   
    save_video(video_frames_60fps, f'output_videos/{output_video}_output_{target_fps}fps.mp4',target_fps)    