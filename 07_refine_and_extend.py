# refine_and_extend_no_ai.py (Cleaned & Corrected Final Version)

import os
import numpy as np
from tqdm import tqdm
import cv2
import json
from collections import deque
from helper.shared_utils import build_reference_data,find_best_match_in_frame
import config

def run_extension_no_ai_final_clean():
    """执行基于模板匹配和双重验证的轨迹补全"""
    

    
    MAX_TEMPLATE_SIZE = 15
    RESIZE_DIM = (64, 128)

    # --- 加载数据 ---
    print("--- Loading data for final extension ---")
    with open(config.FINAL_ROI_LOG, 'r') as f: refined_merge_log = json.load(f)
    with open(config.HISTORY_JSON_PATH, 'r') as f: all_tracks_data = json.load(f)

    staff_full_data = {
        final_id: {int(k): v for original_id in original_ids for k, v in all_tracks_data.get(original_id, {}).items()}
        for final_id, original_ids in refined_merge_log.items()
    }
    
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
                
        # --- c) In-between gap filling ---
        print(f"  Checking for in-between gaps to fill for {staff_id}...")
        sorted_frames_after_extend = sorted(extended_staff_data[staff_id].keys())
        frames_to_add = {}
        for i in range(len(sorted_frames_after_extend) - 1):
            gap_start_frame, gap_end_frame = sorted_frames_after_extend[i], sorted_frames_after_extend[i+1]
            gap = gap_end_frame - gap_start_frame
            if 1 < gap < 75:
                for f_idx in range(gap_start_frame + 1, gap_end_frame):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx); ret, frame = cap.read()
                    if not ret: continue
                    t = (f_idx - gap_start_frame) / gap
                    start_bbox = np.array(extended_staff_data[staff_id][gap_start_frame]); start_center = (start_bbox[:2] + start_bbox[2:]) / 2
                    end_bbox = np.array(extended_staff_data[staff_id][gap_end_frame]); end_center = (end_bbox[:2] + end_bbox[2:]) / 2
                    anchor_center_interp = start_center + (end_center - start_center) * t
                    dummy_prev_bbox = [int(anchor_center_interp[0]-100), int(anchor_center_interp[1]-100), int(anchor_center_interp[0]+100), int(anchor_center_interp[1]+100)]
                    best_match_bbox = find_best_match_in_frame(frame, initial_templates, ref_histogram, dummy_prev_bbox, search_radius=200)
                    if best_match_bbox:
                        frames_to_add[f_idx] = best_match_bbox
        if frames_to_add:
            print(f"  Found and added {len(frames_to_add)} frames in middle gaps for {staff_id}.")
            extended_staff_data[staff_id].update(frames_to_add)
    
    cap.release()
    
    # --- 5. 保存最终产物 ---
    final_data_to_save = {tid: {str(k): v for k, v in sorted(data.items())} for tid, data in extended_staff_data.items()}
    with open(config.STAFF_ID_PATH, 'w') as f:
        json.dump(final_data_to_save, f, indent=4)
    print(f"\nExtension (No AI) complete. Final staff history saved to '{config.STAFF_ID_PATH}'")

if __name__ == "__main__":
    run_extension_no_ai_final_clean()