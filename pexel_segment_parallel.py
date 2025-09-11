import os
import gc
import re
import cv2
import sys
import json
import tqdm
import time
import torch
import shutil
import logging
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import supervision as sv
import torch.multiprocessing as mp
from pycocotools import mask as mask_utils
from tqdm import tqdm

from huggingface_hub import snapshot_download
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

VALIDATION_MODE = True
VISULIZATION_MODE = True
"""
Step 1: Environment settings and model initialization
"""
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s \n>>> %(message)s"
)

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = ("person. bicycle. car. motorcycle. airplane. bus. train. truck. boat. "
"traffic light. fire hydrant. stop sign. parking meter. bench. bird. cat. dog. "
"horse. sheep. cow. elephant. bear. zebra. giraffe. backpack. umbrella. handbag. "
"tie. suitcase. frisbee. skis. snowboard. sports ball. kite. baseball bat. "
"baseball glove. skateboard. surfboard. tennis racket. bottle. wine glass. "
"cup. fork. knife. spoon. bowl. banana. apple. sandwich. orange. broccoli. carrot. "
"hot dog. pizza. donut. cake. chair. couch. potted plant. bed. dining table. toilet. "
"tv. laptop. mouse. remote. keyboard. cell phone. microwave. oven. toaster. sink. "
"refrigerator. book. clock. vase. scissors. teddy bear. hair drier. toothbrush.")

def set_gpu_for_environment(ngpus: int):
    """Set CUDA_VISIBLE_DEVICES for multi-gpu processing"""
    import os
    if ngpus > 1:
        gpu_list = ','.join(str(i) for i in range(ngpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        print(f"Set CUDA_VISIBLE_DEVICES={gpu_list}")

def check_gpu_avaliability():
    """Get GPU detailed information

    Returns:
        gpu_count (int):
            the number of avaliable GPUs
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your PyTorch installation.")
        return 0
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} gpus)")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = (torch.cuda
            .get_device_properties(i).total_memory) / (1024**3)
        print(f"--> GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    return gpu_count

def save_video_mask_as_json(
        images: list,
        annotations: list,
        categories: list,
        save_path: str,
        verbose: bool=False,
):
    """Save the given three list into .json file

    Args:
        images:
            dict list, each item is a dict
            each dict {
                "height": int, attribute of the video
                "width": int, attribute of the video
                "filename": str, path to the video
                "id": int, true frame id
            }
        annotations:
            dict list, each item is a dict
            each dict {
                "id": int, instance id
                "image_id": int, frame id to @images of this mask
                "category_id": int, instance type id
                "segmentation": RLE format mask
                "iscrowd": 1,
                "score": int, the score of SAM 2
            }
        categories:
            dict list, each item is a dict
            each dict {
                "name": str, category name,
                "id": int, category id, start at 1
            }
        
            
    Returns:
        None
    """
    assert save_path is not None, "save_path is None"
    data = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    if verbose:
        logging(f"Data successfully saved to {save_path}")

def load_video_as_tensor(
        video_path: str,
        resolution_limit: bool=True,
        discard_factor: float=None,
        frame_remain_count: int=None,
):
    """Load a video and save it as a tensor.

    Args:
        video_path (str):
            Path to the input video file.
        resolution_limit (bool, optional):
            limit the resolution of video, which
            will resize the maximum dim to 1000 if max(height, width) > 1000,
            or will do nothing if max(height, width) <= 1000
        discard_factor (int, optinal):
            factor to discard frames
        frame_remain_count (int, optinal):
            remain count of frame target
    
    Returns:
        video_tensor (torch.Tensor):
            tensors of video the shape of tensor
            is f * c * w * h, RGB format and [0, 1] values
        frame_idx (list):
            the true index of frames (if discarded some frames,
            use this to locate the true index of frame in original video)
        height (int):
            video frame height
        width (int):
            video frame width
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    frame_idx = []
    frame_idx_counter = 0
    if discard_factor is not None:
        assert 0 < discard_factor < 1, "discard_factor invalid."
        frame_remain_count = int(frame_count * (1 - discard_factor))
    
    if frame_remain_count is not None:
        if frame_remain_count < frame_count:
            step = frame_count / frame_remain_count
            frame_idx = [int(i * step) for i in range(frame_remain_count)]
            # print(f"original {frame_count} frames, remain target {frame_remain_count}")
            # print(f"frame_idx = [{frame_idx}]")
        else:
            frame_idx = [idx for idx in range(frame_count)]
    else:
        frame_idx = [idx for idx in range(frame_count)]

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
        if frame_idx_counter in frame_idx:
            frame = cv2.resize(frame, (resize_width, resize_height))
            # Convert BGR to RGB and then to tensor
            frame = np.ascontiguousarray(frame[:, :, ::-1])
            frame_tensor = torch.from_numpy(frame.transpose(2, 0, 1))
            frames.append(frame_tensor)
        frame_idx_counter += 1

    cap.release()

    # Stack frames into a single tensor
    video_tensor = torch.stack(frames, dim=0) / 255.0
    return video_tensor, frame_idx, resize_height, resize_width

def recover_from_json(json_file_path: str):
    """return the first mask to check the validation"""
    with open(json_file_path, "r") as file:
        data = json.load(file)
    frames = data.get("images", [])
    categories = data.get("categories", [])
    annotations = data.get("annotations", [])
    masks = []
    for anno in annotations:
        rle = {
            "counts": anno["segmentation"]
        }
        frame_id = anno["image_id"]
        rle["size"] = [frames[frame_id]["height"], frames[frame_id]["width"]]
        mask = mask_utils.decode(rle)
        masks.append(mask)
    return masks

def mask_viz(
        video_tensor: torch.Tensor,
        video_segments,
        OBJECTS,
        save_dir: str="./notebooks/mask_viz/"
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
    for frame_idx, segments in video_segments.items():
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
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(save_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)
    print(f">>> Mask saved at {save_dir}")

def get_mask_json(rank, json_save_folder, video_file_list, locks, args):    
    """extract masks from video and save as coco format json file

    Args:
        rank (int):
            the idx of this process
        json_save_folder (str):
            folder to save .json file
        video_file_list (str):
            list of video path
        lock:
            the lock to limit that only one process is in propagating stage
            (or tracking stage) on each GPU, or may cause OOM Error
        args:
            Tuple containing (gpu_id, task_st, task_ed)
    
    Returns:
        None
    """

    gpu_id, task_st, task_ed = args[rank]
    _device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)   # optional but safe
    torch.cuda.set_device(gpu_id)
    torch.autocast(device_type=_device, dtype=torch.bfloat16).__enter__()
    print(f"[Rank {rank}]: using {_device} (physical GPU {gpu_id})")
    print(f"[Rank {rank}]: Model on Device {_device} (GPU {gpu_id})")
    print(f"[Rank {rank}]: task_st == {task_st}, task_ed == {task_ed}")
    print(f"[Rank {rank}]: CUDA visibility = {os.environ['CUDA_VISIBLE_DEVICES']}")
    # Initialize models
    _video_predictor = build_sam2_video_predictor(model_cfg,
                                                  sam2_checkpoint,
                                                  device=_device)
    _sam2_image_model = build_sam2(model_cfg,
                                    sam2_checkpoint, device=_device)
    _image_predictor = SAM2ImagePredictor(_sam2_image_model)

    # init grounding dino model from huggingface
    model_id = "IDEA-Research/grounding-dino-tiny"
    cache_dir = "./local_model_cache"
    local_model_path = snapshot_download(repo_id=model_id, cache_dir=cache_dir)
    _processor = AutoProcessor.from_pretrained(local_model_path,
                                               local_files_only=True)
    _grounding_model = (AutoModelForZeroShotObjectDetection
                        .from_pretrained(local_model_path, local_files_only=True)
                        .to(_device))
    # _processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    # _grounding_model = (AutoModelForZeroShotObjectDetection
    #                     .from_pretrained(model_id, local_files_only=True)
    #                     .to(_device))
    gpu_lock = locks[gpu_id]
    failed_video_counter = 0
    start_time = time.time()
    for task_id in range(task_st, task_ed):

        print(f"[Rank {rank}]: Task processed {task_id - task_st} / {task_ed - task_st} - {(task_id - task_st) / (task_ed - task_st) * 100:.6f}% - Time elapsed {time.time() - start_time:.2f}s")

        video_path = video_file_list[task_id]

        parent_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path)
        video_name = os.path.splitext(video_name)[0]
        json_save_path = os.path.join(json_save_folder,
                                      video_name + "_masks.json")

        print(f"[Rank {rank}]: video_path = {video_path}, mask info save at {json_save_path}")

        video_tensor, frame_idx, height, width = load_video_as_tensor(
            video_path,
            frame_remain_count=300,
        )
        video_tensor = video_tensor.to(_device)
        frame_cnt = len(frame_idx)
        
        # image
        # filename use absolute path instead of relative ones
        json_images = []
        for idx in range(frame_cnt):
            item = {
                "height": height, "width": width,
                "filename": os.path.abspath(video_path),
                "id": frame_idx[idx],
            }
            json_images.append(item)
        
        inference_state = _video_predictor.init_state(video_tensor=video_tensor)

        ann_frame_idx = 0  # the frame index we interact with, default the first frame
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        """
        Step 2: Prompt Grounding DINO and SAM image predictor
        to get the box and mask for specific frame
        """
        # prompt grounding dino to get the box coordinates on specific frame
        image = video_tensor[ann_frame_idx]
        # c * h * w -> h * w * c
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

        # Run Grounding DINO on the first frame
        print(f"[Rank {rank}]: Model on Device: {_device}")
        inputs = _processor(images=image,
                            text=text,
                            return_tensors="pt").to(_device)
        with torch.no_grad():
            outputs = _grounding_model(**inputs)
            results = _processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]],
            )
        del inputs, outputs
        torch.cuda.empty_cache()

        # prompt SAM image predictor to get the mask for the object
        _image_predictor.set_image(np.array(image.convert("RGB")))

        # process the detection results
        input_boxes = results[0]["boxes"].detach().cpu().numpy()
        OBJECTS = [obj for obj in results[0]["labels"]]
        del results  # clear GPU cache
        torch.cuda.empty_cache()

        try:
            # prompt SAM 2 image predictor to get the mask for the object
            masks, scores, logits = _image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
        except AssertionError as e:
            failed_video_counter += 1
            print("-" * 80)
            print(f">>> AssertionError Caught:\n{e}\n")
            print(f"task_id == {task_id}, video_path == {video_path}")
            print(f"failed video counter: {failed_video_counter}")
            print(f"Discarded, skip thie video")
            print("-" * 80)
            # clear GPU memory
            del inference_state
            for var in ["masks", "scores", "logits", "image"]:
                if var in locals():
                    del locals()[var]
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # convert the mask shape to (n, H, W)
        if masks.ndim == 3:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        """
        Step 3: Register each object's positive points to video predictor
        with seperate add_new_points call
        """
        PROMPT_TYPE_FOR_VIDEO = "box"  # or "point"
        assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

        # If you are using point prompts, we uniformly sample positive points based on the mask
        if PROMPT_TYPE_FOR_VIDEO == "point":
            # sample the positive points from mask for each objects
            all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

            for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
                labels = np.ones((points.shape[0]), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = _video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )
        # Using box prompt
        elif PROMPT_TYPE_FOR_VIDEO == "box":
            for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
                _, out_obj_ids, out_mask_logits = _video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    box=box,
                )
        # Using mask prompt is a more straightforward way
        elif PROMPT_TYPE_FOR_VIDEO == "mask":
            for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
                labels = np.ones((1), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = _video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    mask=mask
                )
        else:
            raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")
        
        """
        Step 4: Propagate the video predictor to get the segmentation results for each frame
        """
        video_segments = {}  # video_segments contains the per-frame segmentation results
        gpu_lock.acquire()
        with torch.no_grad():
            for out_frame_idx, out_obj_ids, out_mask_logits in _video_predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            del out_mask_logits
        torch.cuda.empty_cache()
        
        ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
        
        is_viz = (VISULIZATION_MODE and np.random.rand() < 0.1)
        if is_viz:
            print(f"[Rank {rank}] Go into viz_checking mode")
            # visulize the mask to check
            mask_viz(video_tensor, video_segments, OBJECTS)

        """
        Step 5: Prepare other information of json files
        """
        json_categories = []
        json_annotations = []
        # make categories
        label_id = {}
        total_category = 0
        for instance_label_name in OBJECTS:
            if instance_label_name not in label_id:
                total_category += 1
                label_id[instance_label_name] = total_category
        for label_name, id in label_id.items():
            item = {"name": label_name, "id": id}
            json_categories.append(item)
        
        # annotations
        mask_list = []
        is_validate = (VALIDATION_MODE and np.random.rand() < 0.1)
        if is_validate == True:
            print(f"[Rank {rank}] Go into validation mode")
        for idx, segments in video_segments.items():
            for instance_id, mask in segments.items():
                object_category_id = label_id[ID_TO_OBJECTS[instance_id]]
                mask = (mask > 0).squeeze(0).astype(np.uint8)
                if is_validate:
                    mask_list.append(mask)
                rle = mask_utils.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                item = {
                    "id": instance_id, "image_id": idx,
                    "iscrowd": 1, "score": float(scores[instance_id - 1][0]),
                    "category_id": object_category_id,
                    "segmentation": rle["counts"],
                }
                json_annotations.append(item)
        save_video_mask_as_json(json_images, json_annotations, json_categories, json_save_path)
        if is_validate:
            recover_mask_list = recover_from_json(json_save_path)
            for om, rm in zip(mask_list, recover_mask_list):
                assert np.array_equal(om, rm), "Cannot pass the recover check!"
            print(f"[Rank {rank}]: Pass validate checking")
            del mask_list, recover_mask_list
            torch.cuda.empty_cache()
            gc.collect()
                
        del inference_state, video_tensor
        del image, masks, scores, logits, input_boxes
        torch.cuda.empty_cache()
        gc.collect()
        gpu_lock.release()

def launch_parallel_video_processing(
        max_process: int,
        ngpus: int,
        video_folder: str,
        json_save_folder: str,
        start_idx: int,
        end_idx: int,
        task_step: int,
        csv_path: str,
):
    available_gpus = check_gpu_avaliability()
    if available_gpus == 0:
        print("No avaliable GPU. Process Ending...")
        return
    ngpus = min(ngpus, available_gpus)
    print(f"Using {ngpus} GPUs for processing")
    set_gpu_for_environment(ngpus)

    # task list
    video_file_list = []
    if video_folder is not None:
        for parent_dir, subdirs, file_list in os.walk(video_folder):
            for file_name in file_list:
                ext = os.path.splitext(file_name)[-1]
                if ext != ".mp4":
                    continue
                video_path = os.path.join(parent_dir, file_name)
                video_file_list.append(video_path)
    elif csv_path is not None:
        data = pd.read_csv(csv_path, sep=",", index_col=False)
        video_file_list = list(data["file_path"])
        print(f"{len(video_file_list)} items in csv_dataframe")
        if start_idx is not None and end_idx is not None:
            assert start_idx <= end_idx and end_idx <= len(video_file_list)
            video_file_list = video_file_list[start_idx:end_idx:task_step]
    else:
        raise ValueError("No video_folder and csv_path input.")
    finished_task = []
    for filename in os.listdir(json_save_folder):
        if filename.endswith("_masks.json"):
            task_name = os.path.basename(filename)
            task_name = re.sub(r"_masks\.json$", "", task_name)
            finished_task.append(task_name)
    undo_task_list = []
    for file_path in tqdm(video_file_list, desc="Processing Undo Tasks"):
        assert file_path.endswith(".mp4"), "file not ends with .mp4"
        task_name = os.path.basename(file_path)
        task_name = re.sub(r"\.mp4", "", task_name)
        if task_name not in finished_task:
            undo_task_list.append(file_path)
    task_count = len(undo_task_list)
    print(f"[Rank Main]: Found {task_count} tasks left.")

    # split task for each process
    task_per_process = [0] * max_process
    for i in range(task_count):
        task_per_process[i % max_process] += 1
    total = 0
    args_list = []
    for process_idx in range(max_process):
        task_st = total
        task_ed = total + task_per_process[process_idx]
        total += task_per_process[process_idx]
        gpu_id = process_idx % ngpus
        args_list.append((gpu_id, task_st, task_ed))

    manager = mp.Manager()
    locks = [manager.Lock() for _ in range(ngpus)]
    
    print(f"Launch {max_process} processes...")
    start_time = time.time()
    try:
        mp.spawn(get_mask_json,
                 args=(json_save_folder, undo_task_list, locks, args_list),
                 nprocs=max_process)
    except Exception as e:
        print(f"[Rank Main]: An exception is caught: {e}\n" + "-" * 80 + "\n")
    time_cost = time.time() - start_time
    print(f"Spend {time_cost} seconds in total")

def main(max_process: int, ngpus: int,
         video_folder: str, json_save_folder: str,
         start_idx: int, end_idx: int, task_step: int,
         csv_path: str):
    print(f"Starting GPU parallel processing......")
    print(f"process video folder: {video_folder}")
    print(f"json_save_folder: {json_save_folder}")
    print(f"max_process: {max_process}, ngpus: {ngpus}")
    os.makedirs(json_save_folder, exist_ok=True)
    launch_parallel_video_processing(
        max_process, ngpus, video_folder, json_save_folder,
        start_idx, end_idx, task_step, csv_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_process",
        type=int,
        default=1
    )
    parser.add_argument(
        "--ngpus",
        type=int,
        default=1
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        required=False
    )
    parser.add_argument(
        "--json_save_folder",
        type=str,
        required=True
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        required=False
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        required=False
    )
    parser.add_argument(
        "--task_step",
        type=int,
        default=1
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        required=False
    )
    args = parser.parse_args()
    # mp.set_start_method("spawn", force=True)
    main(
        max_process=args.max_process,
        ngpus=args.ngpus,
        video_folder=args.video_folder,
        json_save_folder=args.json_save_folder,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        task_step=args.task_step,
        csv_path=args.csv_path,
    )