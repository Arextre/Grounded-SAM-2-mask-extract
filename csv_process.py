import os
import cv2
import pandas as pd
from warnings import warn
import multiprocessing as mp

VIDEO_FOLDER = "/home/notebook/data/group/huiming/CitySences/"
CSV_PATH = "/home/notebook/data/group/HUIMING/easyanimate_training/human_free_90K.csv"

CSV_SAVE_PATH = "./human_free_90K_processed.csv"

def get_video_durantion(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"video not found: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration

def save_to_csv(csv_info, save_path):
    df = pd.DataFrame(csv_info)
    df.to_csv(save_path, index=False, sep=",")
    print(f"Successfully save csv to {save_path}")

def get_item(file_path):
    file_name = os.path.basename(file_path)
    parent_dir = os.path.dirname(file_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    text_name = file_name_without_ext + ".txt"
    text_path = os.path.join(parent_dir, text_name)

    video_name = file_name
    file_path = os.path.join(parent_dir, file_name)
    if not os.path.exists(text_path):
        warn(f"{file_path} -> .txt file lost")
        return None
    assert os.path.exists(text_path)
    with open(text_path, "r") as f:
        text = f.read()
    start_time_seconds = 0.0
    end_time_seconds = get_video_durantion(file_path)
    video_type = "video"
    video_item = {
        "video_name": video_name,
        "file_path": file_path,
        "text": text,
        "start_time_seconds": start_time_seconds,
        "end_time_seconds": end_time_seconds,
        "type": video_type
    }
    return video_item

def main():
    video_lists = []
    for parent_dir, subdirs, files in os.walk(VIDEO_FOLDER):
        for file_name in files:
            ext = os.path.splitext(file_name)[-1]
            if ext != ".mp4":
                continue
            video_path = os.path.join(parent_dir, file_name)
            video_lists.append(video_path)
    task_count = len(video_lists)
    print(f"Found {task_count} .mp4 files in total")

    csv_info = []
    txt_lost_counter = 0
    with mp.Pool(processes=10) as pool:
        for i, result in enumerate(pool.imap(get_item, video_lists), start=1):
            if result is not None:
                csv_info.append(result)
            else:
                txt_lost_counter += 1
            if i % 10000 == 0:
                print(f">>> Processed {i} videos in total")
    warn(f"{txt_lost_counter} files lost .txt files in total.")
    save_to_csv(csv_info, CSV_SAVE_PATH)

if __name__ == "__main__":
    main()