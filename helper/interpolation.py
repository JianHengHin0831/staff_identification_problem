# refine_and_extend.py (Final version with Cleaning & Interpolation)

import os
import shutil
import torch
import numpy as np
from tqdm import tqdm
import cv2
import json

# 我们不再需要任何AI模型或复杂的匹配函数
from helper.create_annotated_video import linear_interpolate

def run_clean_and_interpolate(staff_full_data):
    """
    最终后处理阶段：
    1. 清洗轨迹，移除位置突变的短片段。
    2. 对清洗后的轨迹中的小空洞进行线性插值。
    """


    # --- 清洗参数 ---
    MAX_JUMP_DISTANCE = 200  # 如果两帧之间的中心点距离超过200像素，视为“瞬移”
    MAX_FRAGMENT_LENGTH = 60 # 如果一个“瞬移”后的片段长度小于60帧，则删除它

    # --- 插值参数 ---
    MAX_INTERPOLATION_GAP = 30 # 只对小于15帧的空洞进行插值

    print("\n--- Stage 2: Cleaning tracks by removing short, disjointed fragments ---")
    cleaned_staff_data = {}
    for staff_id, track_data in staff_full_data.items():
        print(f"  Cleaning track for staff ID: {staff_id}")
        
        sorted_frames = sorted(track_data.keys())
        
        # 将轨迹按“瞬移”点分割成多个片段
        fragments = []
        current_fragment = {}
        
        if sorted_frames:
            current_fragment[sorted_frames[0]] = track_data[sorted_frames[0]]

        
        for i in range(1, len(sorted_frames)):
            prev_frame = sorted_frames[i-1]
            current_frame = sorted_frames[i]
            
            # 只在连续帧之间检查跳变
            if current_frame - prev_frame ==1:
                prev_bbox = track_data[prev_frame]
                current_bbox = track_data[current_frame]
                prev_center = np.array([(prev_bbox[0]+prev_bbox[2])/2, (prev_bbox[1]+prev_bbox[3])/2])
                current_center = np.array([(current_bbox[0]+current_bbox[2])/2, (current_bbox[1]+current_bbox[3])/2])
                dist = np.linalg.norm(current_center - prev_center)
                
                if dist > MAX_JUMP_DISTANCE:
                    # 发现“瞬移”，结束当前片段，开始新片段
                    fragments.append(current_fragment)
                   
                    current_fragment = {}
            
            current_fragment[current_frame] = track_data[current_frame]

        # 添加最后一个片段
        fragments.append(current_fragment)
        
        # 过滤掉过短的片段
        final_track_data = {}
        for frag in fragments:
            if len(frag) > MAX_FRAGMENT_LENGTH:
                final_track_data.update(frag)
            else:
                print(f"    - Removed a fragment of length {len(frag)} due to a large jump.")
                
        cleaned_staff_data[staff_id] = final_track_data

    # --- 4. 【核心逻辑】对清洗后的轨迹进行线性插值 ---
    print("\n--- Stage 3: Interpolating small gaps in cleaned tracks ---")
    interpolated_staff_data = cleaned_staff_data.copy()
    
    for staff_id, track_data in cleaned_staff_data.items():
        if not track_data: continue
        
        sorted_frames = sorted(track_data.keys())
        frames_to_add = {}

        for i in range(len(sorted_frames) - 1):
            gap_start_frame = sorted_frames[i]
            gap_end_frame = sorted_frames[i+1]
            gap = gap_end_frame - gap_start_frame

            if 1 < gap <= MAX_INTERPOLATION_GAP:
                start_bbox = np.array(track_data[gap_start_frame])
                end_bbox = np.array(track_data[gap_end_frame])
                
                for f_idx in range(gap_start_frame + 1, gap_end_frame):
                    t = (f_idx - gap_start_frame) / gap
                    interp_bbox_float = linear_interpolate(start_bbox, end_bbox, t)
                    frames_to_add[f_idx] = [int(coord) for coord in interp_bbox_float]
        
        if frames_to_add:
            print(f"  Interpolated and added {len(frames_to_add)} frames for staff ID {staff_id}.")
            interpolated_staff_data[staff_id].update(frames_to_add)

    # --- 5. 保存最终产物 ---
    return interpolated_staff_data

  
