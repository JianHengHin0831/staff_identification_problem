import cv2
import torch
import os
import shutil
import json
from tqdm import tqdm
import numpy as np 

from helper.create_annotated_video import create_annotated_video
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from boxmot.trackers.bytetrack.bytetrack import ByteTrack
import config

def run_tracker():
    # --- 1. config ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    YOLO_MODEL_PATH = 'yolov8l.pt'

    print(f"Using device: {DEVICE}")

    # --- 2. initialize model ---
    print("Initializing SAHI-wrapped YOLOv8 model for detection...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=YOLO_MODEL_PATH,
        confidence_threshold=0.25, 
        device=DEVICE,
    )
    
    print("Initializing BYTETracker...")
    tracker = ByteTrack(
        track_thresh=0.1,      # filter rate
        track_buffer=50,       # track buffer that the bytetrack can wait
        match_thresh=0.9,      # IOU match threshold
        frame_rate=30          # 30 frame/s
    )

    # --- 3. tracker DIR & VIDEO preparation ---
    if os.path.exists(config.TRACKER_ROI_DIR): shutil.rmtree(config.TRACKER_ROI_DIR)
    os.makedirs(config.TRACKER_ROI_DIR)

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- 4. main loop ---
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
            
            # turn Sahi result to the bytetrack input (N, 6): [x1, y1, x2, y2, score, class_id]
            detections_for_bytetrack = []
            for pred in sahi_result.object_prediction_list:
                if pred.category.id == 0: # filter 'person'
                    bbox = pred.bbox
                    x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
                    confidence = pred.score.value
                    detections_for_bytetrack.append([x1, y1, x2, y2, confidence, 0])
            
            if len(detections_for_bytetrack) > 0:
                detections_np = np.array(detections_for_bytetrack)
            else:
                detections_np = np.empty((0, 6))

            # update ByteTrack, .update() return (M, 8) numpy list: [x1, y1, x2, y2, track_id, score, class_id]
            online_targets = tracker.update(detections_np, frame)

            for target in online_targets:
                x1, y1, x2, y2, track_id, _, _, _ = target
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
                    track_dir = os.path.join(config.TRACKER_ROI_DIR, str(folder_id))
                    os.makedirs(track_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(track_dir, f"frame_{str(frame_idx).zfill(5)}.jpg"), roi)

            pbar.update(1)

    # --- 5. release resource ---
    cap.release()
    print("\nProcessing finished.")
    
    with open(config.HISTORY_JSON_PATH, 'w') as f:
        json.dump(all_tracks_data, f, indent=4)
    print(f"Tracks history saved to '{config.HISTORY_JSON_PATH}'.")

    create_annotated_video(
        tracks_history_path=config.HISTORY_JSON_PATH,
        output_video_path=config.TRACKER_VID
    )

if __name__ == "__main__":
    run_tracker()