import os
import re
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
import supervision as sv
import torch.multiprocessing as mp
from einops import rearrange
import torch.nn.functional as F
from torchvision.io import read_video
from pycocotools import mask as mask_utils

JSON_SAVE_FOLDER = "/home/notebook/code/personal/S9060429/Grounded-SAM-2/notebooks/json_folder"
MASK_VIZ_FOLDER = "/home/notebook/code/personal/S9060429/Grounded-SAM-2/notebooks/mask_viz_test"

DEVICE = "cuda:0"
torch.cuda.set_device(DEVICE)

def mask_viz(
        video_tensor: torch.Tensor,
        video_segments: dict,
        ID_TO_OBJECTS: dict,
        save_dir: str,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for frame_idx, segments in tqdm(video_segments.items(),
                                    desc="Frames visulizing"):
        # c * h * w -> h * w * c
        img = video_tensor[frame_idx].permute(1, 2, 0).cpu().numpy()
        # RGB -> BGR
        img = np.ascontiguousarray(img[:, :, ::-1])
        img = (img * 255).astype(np.uint8)
        
        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)
        
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
            mask=masks, # (n, h, w)
            class_id=np.array(object_ids, dtype=np.int32),
        )
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(),
                                                 detections=detections)
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(
            annotated_frame, 
            detections=detections,
            labels=[ID_TO_OBJECTS[i] for i in object_ids],
        )
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame,
                                                  detections=detections)
        cv2.imwrite(
            os.path.join(save_dir, f"annotated_frame_{frame_idx:05d}.jpg"),
            annotated_frame,
        )
    print(f">>> Mask viz saved at folder: {save_dir}")


def load_video_as_tensor(
        video_path: str,
        video_height: int,
        video_width: int,
        frame_id: list,
):
    video_tensor, _, _ = read_video(video_path,
                                    pts_unit="src",
                                    output_format="TCHW")
    video_tensor = video_tensor.to(device=DEVICE, dtype=torch.float32)
    valid_frame_mask = [(idx in frame_id) for idx in range(len(video_tensor))]
    video_tensor = video_tensor[valid_frame_mask]
    assert len(video_tensor) == len(frame_id)
    video_tensor = F.interpolate(
        video_tensor,
        size=(video_height, video_width),
        mode="bilinear",
        align_corners=False
    )
    if torch.max(video_tensor) > 1:
        video_tensor /= 255.0
    return video_tensor

def load_json_file(json_file_path: str):
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)
    frame_info = data.get("images", [])
    mask_info = data.get("annotations", [])
    category_info = data.get("categories", [])
    assert len(frame_info) > 0
    video_path = frame_info[0]["filename"]
    video_height = frame_info[0]["height"]
    video_width = frame_info[0]["width"]
    frame_id = [frame["id"] for frame in frame_info]
    video_tensor = load_video_as_tensor(video_path,
                                        video_height,
                                        video_width,
                                        frame_id)
    video_segments = {}
    CATEGORY_ID_TO_CATEGORY = {}
    for cat_item in category_info:
        CATEGORY_ID_TO_CATEGORY[cat_item["id"]] = cat_item["name"]
    ID_TO_OBJECTS = {}
    for anno_item in mask_info:
        frame_idx = anno_item["image_id"]
        rle = {
            "counts": anno_item["segmentation"],
            "size": [video_height, video_width],
        }
        mask = mask_utils.decode(rle).astype(np.uint8)
        mask = mask[None, :, :]
        instance_id = anno_item["id"]
        if frame_idx not in video_segments:
            video_segments[frame_idx] = {}
        video_segments[frame_idx][instance_id] = mask
        if instance_id not in ID_TO_OBJECTS:
            category_id = anno_item["category_id"]
            ID_TO_OBJECTS[instance_id] = CATEGORY_ID_TO_CATEGORY[category_id]
    return video_tensor, video_segments, ID_TO_OBJECTS

def pipeline(json_file_path: str):
    video_tensor, video_segments, ID_TO_OBJECTS = load_json_file(json_file_path)
    file_name = os.path.basename(json_file_path)
    assert file_name.endswith("_masks.json")
    file_name = re.sub(r"_masks\.json$", "", file_name)
    save_dir = os.path.join(MASK_VIZ_FOLDER, file_name)
    mask_viz(video_tensor, video_segments, ID_TO_OBJECTS, save_dir)
    return 1

def main():
    json_file_list = [
        os.path.join(JSON_SAVE_FOLDER, file_name)
        for file_name in os.listdir(JSON_SAVE_FOLDER)
        if file_name.endswith(".json")
    ]
    task_counter = len(json_file_list)
    print(f"Found {task_counter} .json files in total.")

    with mp.Pool(processes=3) as pool:
        with tqdm(total=task_counter, desc="Processed JSON files") as pbar:
            for _ in pool.imap_unordered(pipeline, json_file_list):
                pbar.update(1)

    # for json_file in tqdm(json_file_list, desc="JSON file processing"):


if __name__ == "__main__":
    main()