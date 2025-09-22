# refine_and_extend_no_ai.py (Cleaned & Corrected Final Version)

import os
import numpy as np
from tqdm import tqdm
import cv2
import json
from collections import deque
from helper.shared_utils import build_reference_data,find_best_match_in_frame
import config
from helper.create_annotated_video import create_annotated_video
from helper.interpolation import run_clean_and_interpolate

def staff_tracking():
    MAX_TEMPLATE_SIZE = 15
    RESIZE_DIM = (64, 128)

    # Loading data
    print("--- Loading data for final extension ---")
    with open(config.FINAL_ROI_LOG, 'r') as f: refined_merge_log = json.load(f)
    with open(config.HISTORY_JSON_PATH, 'r') as f: all_tracks_data = json.load(f)

    with open(config.STAFF_LIST_PATH, "r") as f:
        staff_list = json.load(f)["staff_ids"]

    full_data = {
        final_id: {int(k): v for original_id in original_ids for k, v in all_tracks_data.get(original_id, {}).items()}
        for final_id, original_ids in refined_merge_log.items()
    }

    staff_full_data = {fid: data for fid, data in full_data.items() if fid in staff_list}
    
    extended_staff_data = staff_full_data.copy()
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for staff_id, track_data in staff_full_data.items():
        print(f"\nExtending track for staff ID: {staff_id}")
        
        roi_folder = os.path.join(config.FINAL_ROI_DIR, staff_id)
        initial_templates, ref_histogram = build_reference_data(roi_folder, num_templates=MAX_TEMPLATE_SIZE, resize_dim=RESIZE_DIM)
        if not initial_templates or ref_histogram is None:
            print(f"Warning: Could not build reference data for {staff_id}. Skipping.")
            continue

        sorted_frames = sorted(track_data.keys())
        start_frame, end_frame = sorted_frames[0], sorted_frames[-1]

        # --- a) Backtracking ---
        templates_deque = deque(initial_templates, maxlen=MAX_TEMPLATE_SIZE)
        for f_idx in tqdm(range(start_frame - 1, -1, -1), desc=f"  Backtracking {staff_id}"):
            prev_bbox = extended_staff_data[staff_id].get(f_idx + 1)
            if not prev_bbox: break
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx); ret, frame = cap.read()
            if not ret: break
            best_match_bbox = find_best_match_in_frame(frame, list(templates_deque), ref_histogram, prev_bbox)
            if best_match_bbox:
                extended_staff_data[staff_id][f_idx] = best_match_bbox
                x1, y1, x2, y2 = best_match_bbox
                new_template_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                templates_deque.appendleft(cv2.resize(new_template_gray, RESIZE_DIM))
            else:
                break

        # --- b) Forward tracking ---
        templates_deque = deque(initial_templates, maxlen=MAX_TEMPLATE_SIZE)
        for f_idx in tqdm(range(end_frame + 1, total_frames), desc=f"  Forward tracking {staff_id}"):
            prev_bbox = extended_staff_data[staff_id].get(f_idx - 1)
            if not prev_bbox: break 
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx); ret, frame = cap.read()
            if not ret: break
            best_match_bbox = find_best_match_in_frame(frame, list(templates_deque), ref_histogram, prev_bbox)
            if best_match_bbox:
                extended_staff_data[staff_id][f_idx] = best_match_bbox
                x1, y1, x2, y2 = best_match_bbox
                new_template_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                templates_deque.append(cv2.resize(new_template_gray, RESIZE_DIM))
            else:
                break

    cap.release()
    
    # c) in between interpolation
    interpolated_staff_data = run_clean_and_interpolate(extended_staff_data.copy())
    
    # output result
    final_data_to_save = {tid: {str(k): v for k, v in sorted(data.items())} for tid, data in interpolated_staff_data.items()}
    
    with open(config.STAFF_ID_PATH, 'w') as f:
        json.dump(final_data_to_save, f, indent=4)
    print(f"\nExtension (No AI) complete. Final staff history saved to '{config.STAFF_ID_PATH}'")

    create_annotated_video(
        tracks_history_path=config.STAFF_ID_PATH,
        output_video_path=config.TRACKING_VID
    )

if __name__ == "__main__":
    staff_tracking()