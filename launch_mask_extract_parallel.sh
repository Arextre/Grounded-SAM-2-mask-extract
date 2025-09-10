if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <ngpus> <max_process>"
    exit 1
fi

NGPUS=$1
MAX_PROCESS=$2
VIDEO_FOLDER="./notebooks/videos/mouth-closed/"
JSON_SAVE_FOLDER="./notebooks/json_folder/"

export TRANSFORMERS_OFFLINE=1

HF_HUB_OFFLINE=1 \
/home/notebook/code/personal/S9060429/.venv/bin/python pexel_segment_parallel.py \
    --max_process=$MAX_PROCESS \
    --ngpus=$NGPUS \
    --video_folder=$VIDEO_FOLDER \
    --json_save_folder=$JSON_SAVE_FOLDER \
    --csv_path="./human_free_27K_processed.csv" \
    --start_idx=0 \
    --end_idx=50 \
    --task_step=5