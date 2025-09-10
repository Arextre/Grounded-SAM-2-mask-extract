import cv2
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.io import read_video

VIDEO_PATH = "/home/notebook/code/personal/S9060429/Grounded-SAM-2/notebooks/videos/mouth-closed/01.mp4"
DEVICE = "cuda:0"

torch.cuda.set_device(device=DEVICE)
torch.autocast(device_type=DEVICE, dtype=torch.float32).__enter__()

def th_load_video(
        video_path: str,
        resolution_limit: bool=True,
        discard_interval: int=None,
):
    video_tensor, _, _ = read_video(video_path,
                                             pts_unit="sec",
                                             output_format="TCHW")
    video_tensor = video_tensor.to(device=DEVICE, dtype=torch.float32)
    if torch.max(video_tensor) > 1:
        video_tensor = video_tensor / 255.0
    video_height = video_tensor.shape[2]
    video_width = video_tensor.shape[3]
    video_frame_count = video_tensor.shape[0]
    reserved_frame_mask = [(discard_interval is None
                            or (i + 1) % discard_interval != 0)
                           for i in range(video_frame_count)]
    frame_idx = [i for i in range(video_frame_count) if reserved_frame_mask[i]]
    video_tensor = video_tensor[reserved_frame_mask]
    if resolution_limit and max(video_width, video_height) > 1000:
        resize_scale = 1000 / max(video_width, video_height)
        video_width = int(video_width * resize_scale)
        video_height = int(video_height * resize_scale)
        video_tensor = F.interpolate(
            video_tensor,
            size=(video_height, video_width),
            mode="bilinear",
            align_corners=False,
        )
    return video_tensor, frame_idx, video_height, video_width

def cv_load_video(
        video_path: str,
        resolution_limit: bool=True,
        discard_interval: int=None,
):
    """Read a video and save it as a tensor.

    Args:
        video_path (str): Path to the input video file.
        save_path (str): Path to save the output tensor.
        resolution_limit (bool, optional): limit the resolution of video, which
            will resize the maximum dim to 1000 if max(height, width) > 1000,
            or will do nothing if max(height, width) <= 1000
        discard_interval (int, optinal): interval to discard some frames

    Returns:
        video_tensor (torch.Tensor): tensors of video
            the shape of tensor is f * c * w * h, RGB format and [0, 1] values
        frame_idx (list): the true index of frames (if discarded some frames,
            use this to locate the true index of frame in original video)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    frames = []
    frame_idx = []
    frame_idx_counter = 0

    resize_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    resize_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if resolution_limit and max(resize_width, resize_height) > 1000:
        resize_scale = 1000 / max(resize_width, resize_height)
        resize_width = int(resize_scale * resize_width)
        resize_height = int(resize_scale * resize_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (discard_interval is None
            or (frame_idx_counter + 1) % discard_interval != 0):
            frame = cv2.resize(frame, (resize_width, resize_height))
            # Convert BGR to RGB and then to tensor
            frame = np.ascontiguousarray(frame[:, :, ::-1])
            frame_tensor = torch.from_numpy(frame.transpose(2, 0, 1))
            frames.append(frame_tensor)
            frame_idx.append(frame_idx_counter)
        frame_idx_counter += 1

    cap.release()

    # Stack frames into a single tensor
    video_tensor = (torch.stack(frames, dim=0)
                         .to(device=DEVICE, dtype=torch.float32)) / 255.0
    return video_tensor, frame_idx, resize_height, resize_width

def main():
    ITERATE_TIME = 1

    th_start = time.time()
    for _ in tqdm(range(ITERATE_TIME), desc="torch loading progress"):
        th_result = th_load_video(VIDEO_PATH, discard_interval=2)
    th_end = time.time()

    cv_start = time.time()
    for _ in tqdm(range(ITERATE_TIME), desc="cv2 loading progress"):
        cv_result = cv_load_video(VIDEO_PATH, discard_interval=2)
    cv_end = time.time()
    print(f"torch time cost: {th_end - th_start: .5f}")
    print(f"cv2   time cost: {cv_end - cv_start: .5f}")
    assert (torch.allclose(th_result[0],
                          cv_result[0],
                          rtol=1e-5,
                          atol=5e-3,
                          equal_nan=False)
            and th_result[1:] == cv_result[1:])

if __name__ == "__main__":
    main()