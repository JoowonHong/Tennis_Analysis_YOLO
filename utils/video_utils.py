import cv2


def read_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)  
    frames = []
    # FPS 읽기
    fps = read_fps(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)           
    cap.release()
    
    return frames,fps



def save_video(output_video_frame, output_video_path ,fps):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_video_frame[0].shape[1], output_video_frame[0].shape[0]))
    for frame in output_video_frame:
        out.write(frame)
    out.release()
    print(f"Video saved at {output_video_path}")


