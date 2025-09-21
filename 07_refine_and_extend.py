# refine_and_extend_no_ai.py (Cleaned & Corrected Final Version)

import os
import shutil
import numpy as np
from tqdm import tqdm
import cv2
import json
from collections import deque

# --- 1. 辅助函数 ---

def calculate_histogram(image_bgr, bins=[8, 8, 8]):
    """计算BGR图像的3D颜色直方图并归一化"""
    if image_bgr is None or image_bgr.size == 0: return None
    hist = cv2.calcHist([image_bgr], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist

def build_reference_data(roi_folder, num_templates=10, resize_dim=(64, 128)):
    """从一个ROI文件夹中创建模板库和平均颜色直方图指纹"""
    templates = []
    hists = []
    image_files = []
    for root, _, files in os.walk(roi_folder):
        for file in files:
            if file.endswith('.jpg'):
                image_files.append(os.path.join(root, file))
    
    if not image_files: return [], None

    sample_size = min(len(image_files), num_templates)
    sample_files = np.random.choice(image_files, sample_size, replace=False)

    for img_path in sample_files:
        img = cv2.imread(img_path)
        if img is not None:
            hists.append(calculate_histogram(img))
            template_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            templates.append(cv2.resize(template_gray, resize_dim))
            
    if not hists: return templates, None
    
    valid_hists = [h for h in hists if h is not None]
    if not valid_hists: return templates, None
    
    return templates, np.mean(valid_hists, axis=0)

def find_best_match_in_frame(frame, templates, ref_histogram, prev_bbox, 
                             search_radius=200, match_threshold=0.6, hist_threshold=0.5):
    """带颜色直方图双重验证的、多尺度的模板匹配"""
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    px, py = (prev_bbox[0] + prev_bbox[2]) // 2, (prev_bbox[1] + prev_bbox[3]) // 2
    x_min = max(0, px - search_radius); y_min = max(0, py - search_radius)
    x_max = min(frame.shape[1], px + search_radius); y_max = min(frame.shape[0], py + search_radius)
    
    search_area_gray = frame_gray[y_min:y_max, x_min:x_max]
    # 【KEPT & EXPLAINED】我们保留这个变量，用于后续高效的颜色ROI裁剪
    search_area_color = frame[y_min:y_max, x_min:x_max]

    if search_area_gray.size == 0: return None
    
    best_overall_score = -1
    final_best_bbox = None

    for template in templates:
        for scale in np.linspace(0.8, 1.2, 5):
            resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
            th, tw = resized_template.shape
            
            if th > search_area_gray.shape[0] or tw > search_area_gray.shape[1]: continue

            res = cv2.matchTemplate(search_area_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > match_threshold:
                top_left_local = max_loc
                
                # 【核心修正】现在我们正确地从更小的 `search_area_color` 中裁剪
                candidate_roi_color = search_area_color[
                    top_left_local[1] : top_left_local[1] + th,
                    top_left_local[0] : top_left_local[0] + tw
                ]

                if candidate_roi_color.size > 0:
                    candidate_hist = calculate_histogram(candidate_roi_color)
                    if candidate_hist is None: continue
                    hist_similarity = cv2.compareHist(ref_histogram, candidate_hist, cv2.HISTCMP_CORREL)

                    if hist_similarity >= hist_threshold:
                        if max_val > best_overall_score:
                            best_overall_score = max_val
                            final_best_bbox = [
                                top_left_local[0] + x_min, top_left_local[1] + y_min,
                                top_left_local[0] + x_min + tw, top_left_local[1] + y_min + th
                            ]
    return final_best_bbox

# --- 2. 主函数 ---

def run_extension_no_ai_final_clean():
    """执行基于模板匹配和双重验证的轨迹补全"""
    # --- 配置 ---
    REFINED_ROI_DIR = "tracked_rois_refined"
    TRACKS_HISTORY_PATH = "tracks_history.json"
    MERGE_LOG_PATH = "merge_log_refined.json"
    VIDEO_PATH = "sample.mp4"
    EXTENDED_STAFF_HISTORY_PATH = "staff_history_extended_no_ai.json"
    
    MAX_TEMPLATE_SIZE = 15
    RESIZE_DIM = (64, 128)

    # --- 加载数据 ---
    print("--- Loading data for final extension ---")
    with open(MERGE_LOG_PATH, 'r') as f: refined_merge_log = json.load(f)
    with open(TRACKS_HISTORY_PATH, 'r') as f: all_tracks_data = json.load(f)

    staff_full_data = {
        final_id: {int(k): v for original_id in original_ids for k, v in all_tracks_data.get(original_id, {}).items()}
        for final_id, original_ids in refined_merge_log.items()
    }
    
    extended_staff_data = staff_full_data.copy()
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for staff_id, track_data in staff_full_data.items():
        print(f"\nExtending track for staff ID: {staff_id}")
        
        roi_folder = os.path.join(REFINED_ROI_DIR, staff_id)
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
    with open(EXTENDED_STAFF_HISTORY_PATH, 'w') as f:
        json.dump(final_data_to_save, f, indent=4)
    print(f"\nExtension (No AI) complete. Final staff history saved to '{EXTENDED_STAFF_HISTORY_PATH}'")

if __name__ == "__main__":
    run_extension_no_ai_final_clean()