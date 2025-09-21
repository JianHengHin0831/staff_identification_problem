# run_tracker.py (Final & Cleanest version: SAHI + boxmot.BYTETracker)

import cv2
import torch
import time
import os
import shutil
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path

from helper.shared_utils import create_debug_video
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# 【核心】导入独立的ByteTrack
from boxmot.trackers.bytetrack.bytetrack import ByteTrack

def run_tracking_sahi_bytetrack():
    """
    终极版：使用SAHI进行检测，使用独立的ByteTrack库进行追踪。
    """
    # --- 1. 配置 ---
    VIDEO_PATH = "sample.mp4"
    OUTPUT_ROI_DIR = "tracked_rois_initial"
    OUTPUT_JSON_PATH = "tracks_history.json"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    YOLO_MODEL_PATH = 'yolov8l.pt'

    print(f"Using device: {DEVICE}")

    # --- 2. 初始化模型 ---
    print("Initializing SAHI-wrapped YOLOv8 model for detection...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=YOLO_MODEL_PATH,
        confidence_threshold=0.25, # ByteTrack会处理低分框，所以这个值可以设得比较高
        device=DEVICE,
    )
    
    print("Initializing BYTETracker...")
    # 初始化ByteTrack追踪器
    tracker = ByteTrack(
        track_thresh=0.25,     # 过滤掉低于此置信度的检测框（ByteTrack的第一阶段）
        track_buffer=50,       # 轨迹可以“失踪”的最大帧数
        match_thresh=0.8,      # IOU匹配阈值
        frame_rate=30          # 视频帧率
    )

    # --- 3. 准备目录和视频读取器 ---
    if os.path.exists(OUTPUT_ROI_DIR): shutil.rmtree(OUTPUT_ROI_DIR)
    os.makedirs(OUTPUT_ROI_DIR)

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- 4. 主处理循环 ---
    all_tracks_data = {}
    tracker_id_to_folder_id = {}
    
    with tqdm(total=total_frames, desc="Tracking with SAHI+ByteTrack") as pbar:
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sahi_result = get_sliced_prediction(
                frame_rgb, detection_model,
                slice_height=640, slice_width=640,
                overlap_height_ratio=0.2, overlap_width_ratio=0.2
            )
            
            # 将SAHI结果转换为ByteTrack需要的 (N, 6) numpy数组: [x1, y1, x2, y2, score, class_id]
            detections_for_bytetrack = []
            for pred in sahi_result.object_prediction_list:
                if pred.category.id == 0: # 筛选'person'
                    bbox = pred.bbox
                    x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
                    confidence = pred.score.value
                    detections_for_bytetrack.append([x1, y1, x2, y2, confidence, 0])
            
            if len(detections_for_bytetrack) > 0:
                detections_np = np.array(detections_for_bytetrack)
            else:
                detections_np = np.empty((0, 6))

            # 更新ByteTrack追踪器, .update()返回一个(M, 7)的numpy数组: [x1, y1, x2, y2, track_id, score, class_id]
            online_targets = tracker.update(detections_np, frame)

            for target in online_targets:
                x1, y1, x2, y2, track_id, score, cls = target
                original_track_id = int(track_id)
                
                if original_track_id not in tracker_id_to_folder_id:
                    folder_id = f"track_{str(frame_idx).zfill(5)}"
                    tracker_id_to_folder_id[original_track_id] = folder_id
                    print(f"New ByteTrack detected: ID {original_track_id} -> Folder '{folder_id}'")
                
                folder_id = tracker_id_to_folder_id[original_track_id]
                bbox = list(map(int, [x1, y1, x2, y2]))
                
                if folder_id not in all_tracks_data: all_tracks_data[folder_id] = {}
                all_tracks_data[folder_id][frame_idx] = bbox

                roi = frame[max(0, bbox[1]):min(frame.shape[0], bbox[3]), max(0, bbox[0]):min(frame.shape[1], bbox[2])]
                if roi.size > 0:
                    track_dir = os.path.join(OUTPUT_ROI_DIR, str(folder_id))
                    os.makedirs(track_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(track_dir, f"frame_{str(frame_idx).zfill(5)}.jpg"), roi)

            pbar.update(1)

    # --- 5. 释放资源和保存结果 ---
    cap.release()
    print("\nProcessing finished.")
    
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(all_tracks_data, f, indent=4)
    print(f"Tracks history saved to '{OUTPUT_JSON_PATH}'.")

    create_debug_video(
        original_video_path=VIDEO_PATH,
        tracks_history_path=OUTPUT_JSON_PATH,
        output_video_path="debug_tracker_output_sahi_bytetrack.mp4"
    )

if __name__ == "__main__":
    run_tracking_sahi_bytetrack()