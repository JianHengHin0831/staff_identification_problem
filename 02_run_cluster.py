# run_cluster.py

import os
import shutil
import torch
import numpy as np
from tqdm import tqdm
import cv2
import json
from helper.create_annotated_video import create_annotated_video

# 从共享文件中导入辅助函数
from helper.shared_utils import get_feature_extractor, get_preprocess_transform, extract_features


def find_closest_transition_frames(roi_dir, track_id1, track_id2, intervals1, intervals2):
    min_gap = float('inf')
    best_frame1 = -1
    best_frame2 = -1
    id_of_best_frame1 = None
    id_of_best_frame2 = None

    # 遍历所有时间段对，寻找最小的时间间隔
    for start1, end1 in intervals1:
        for start2, end2 in intervals2:
            # 比较 end1 和 start2
            gap1 = abs(start2 - end1)
            if gap1 < min_gap:
                min_gap = gap1
                best_frame1, best_frame2 = end1, start2
                id_of_best_frame1, id_of_best_frame2 = track_id1, track_id2

            # 比较 end2 和 start1
            gap2 = abs(start1 - end2)
            if gap2 < min_gap:
                min_gap = gap2
                best_frame1, best_frame2 = end2, start1 # 注意顺序
                id_of_best_frame1, id_of_best_frame2 = track_id2, track_id1
                
    if best_frame1 == -1 or best_frame2 == -1:
        return None, None, None, None, float('inf')

    path1 = os.path.join(roi_dir, id_of_best_frame1, f"frame_{str(best_frame1).zfill(5)}.jpg")
    path2 = os.path.join(roi_dir, id_of_best_frame2, f"frame_{str(best_frame2).zfill(5)}.jpg")

    # 返回所有需要的信息：路径1, 路径2, 帧号1, 帧号2, 最小间隔
    return path1, path2, best_frame1, best_frame2, min_gap

def get_track_time_range(roi_dir, track_id):
    """从文件名解析轨迹的开始和结束帧"""
    track_path = os.path.join(roi_dir, track_id)
    image_files = [f for f in os.listdir(track_path) if f.endswith('.jpg')]
    if not image_files:
        return -1, -1
    
    # 解析文件名中的数字
    frame_numbers = [int(f.replace('frame_', '').replace('.jpg', '')) for f in image_files]
    return min(frame_numbers), max(frame_numbers)

def build_gallery(roi_dir, track_id, extractor, preprocessor, device, sample_size=15):
    """为每个轨迹构建一个特征画廊"""
    gallery_features = []
    track_path = os.path.join(roi_dir, track_id)
    image_files = [f for f in os.listdir(track_path) if f.endswith('.jpg')]
    
    if not image_files: return None
    
    sample_files = np.random.choice(image_files, min(len(image_files), sample_size), replace=False)
    
    for fname in sample_files:
        img_path = os.path.join(track_path, fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is not None:
            features = extract_features(extractor, preprocessor, img_bgr, device)
            if features is not None:
                gallery_features.append(features)

    if not gallery_features: return None
    return torch.cat(gallery_features, dim=0)

def get_track_time_intervals(roi_dir, track_id, max_gap=10):
    track_path = os.path.join(roi_dir, track_id)
    image_files = [f for f in os.listdir(track_path) if f.endswith('.jpg')]
    if not image_files:
        return []
    
    frame_numbers = sorted([int(f.replace('frame_', '').replace('.jpg', '')) for f in image_files])
    
    if not frame_numbers:
        return []

    intervals = []
    start_interval = frame_numbers[0]
    for i in range(1, len(frame_numbers)):
        # 如果当前帧号与上一帧号的间隔大于max_gap，则认为是一个新的时间段
        if frame_numbers[i] - frame_numbers[i-1] > max_gap:
            intervals.append((start_interval, frame_numbers[i-1]))
            start_interval = frame_numbers[i]
    # 添加最后一个时间段
    intervals.append((start_interval, frame_numbers[-1]))
    
    return intervals

def check_intervals_overlap(intervals1, intervals2):
    """检查两组时间段列表是否有任何重叠"""
    for start1, end1 in intervals1:
        for start2, end2 in intervals2:
            # 判断两个区间 [start1, end1] 和 [start2, end2] 是否重叠
            if max(start1, start2) <= min(end1, end2):
                return True
    return False

def get_min_time_gap(intervals1, intervals2):
    """计算两组时间段之间的最小间隔"""
    min_gap = float('inf')
    for start1, end1 in intervals1:
        for start2, end2 in intervals2:
            gap = max(start1 - end2, start2 - end1)
            if gap < min_gap:
                min_gap = gap
    return min_gap

def calculate_spatiotemporal_score(
    roi_dir, track_id1, track_id2, 
    gallery1, gallery2, intervals1, intervals2,
    tracks_history,
    merge_log,
    extractor, preprocessor, device):
    """【核心改进】使用时间段列表计算分数"""
    # 1. 检查时间重叠
    if check_intervals_overlap(intervals1, intervals2):
        return 0.0, "Time Overlap"

    # 2. 计算最小时间间隔和时间因子
    min_gap = get_min_time_gap(intervals1, intervals2)
    
    reason_time=''
    # 【建议的平滑函数】
    if min_gap <= 30:
        factor_time = 1.25 # 最高奖励
        reason_time = f"Very Close Gap ({min_gap}f)"
    elif min_gap <= 300: # 将平滑衰减的窗口拉长
        # 从 1.15 线性衰减到 0.9
        # 衰减公式: start_value - (current_gap - window_start) * (total_decay / window_size)
        factor_time = 1.15 - (min_gap - 30) * (0.25 / 270)
        reason_time = f"Medium Gap ({min_gap}f)"
    else: # 间隔非常大 (> 10秒)
        factor_time = 0.9 # 最大惩罚
        reason_time = f"Large Gap ({min_gap}f)"
    # 3. 计算外观相似度
    sim_global = min(
        torch.mm(gallery1, gallery2.T).max(dim=1).values.mean().item(),
        torch.mm(gallery1, gallery2.T).max(dim=0).values.mean().item()
    )

    # b) 【核心修正】衔接点相似度 (sim_transition)
    # 使用新的、严谨的函数来查找衔接帧
    #path1, path2 = find_closest_transition_frames(roi_dir, track_id1, track_id2, intervals1, intervals2)
    path1, path2, frame1, frame2, min_gap = find_closest_transition_frames(
        roi_dir, track_id1, track_id2, intervals1, intervals2
    )
    sim_transition = 0.0
    if path1 and path2 and os.path.exists(path1) and os.path.exists(path2):
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        if img1 is not None and img2 is not None:
            feat1 = extract_features(extractor, preprocessor, img1, device)
            feat2 = extract_features(extractor, preprocessor, img2, device)
            if feat1 is not None and feat2 is not None:
                sim_transition = torch.mm(feat1, feat2.T).item()
    else:
        # 如果找不到衔接帧（理论上不应发生，除非轨迹为空），则衔接相似度为0
        sim_transition = 0.0
            
    # 4. 动态加权融合 (保持不变)
    if min_gap <= 45: 
        w_global, w_transition = 0.4, 0.6
    else:
        w_global, w_transition = 0.7, 0.3
        
    sim_appearance = w_global * sim_global + w_transition * sim_transition
    reason_appearance = f"Global Appearance is {sim_global} Local Appearance is {sim_transition}"

     # --- 3. 空间维度 (Spatial) ---
    score_spatial = 0.0
    dist = float('inf')
    bbox1, bbox2 = None, None
    
    # 从路径中解析出所属的聚类ID
    cluster_id_for_frame1 = os.path.basename(os.path.dirname(path1))
    cluster_id_for_frame2 = os.path.basename(os.path.dirname(path2))

    for orig_id in merge_log[cluster_id_for_frame1]:
        if str(frame1) in tracks_history.get(orig_id, {}):
            bbox1 = tracks_history[orig_id][str(frame1)]
            break
    for orig_id in merge_log[cluster_id_for_frame2]:
        if str(frame2) in tracks_history.get(orig_id, {}):
            bbox2 = tracks_history[orig_id][str(frame2)]
            break
            
    if bbox1 and bbox2:
        center1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
        center2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
        dist = np.linalg.norm(center1 - center2)
        score_spatial = np.exp(-0.003 * dist)
    
    # --- 4. 最终分数计算 ---
    # 权重分配：外观占70%，空间占30%
    w_appearance = 0.85
    w_spatial = 0.15
    reason_spatial=f"Spatial:  {score_spatial}"
    
    combined_score = (w_appearance * sim_appearance + w_spatial * score_spatial)
    final_score = combined_score * factor_time
    reason = reason_time+reason_appearance+reason_spatial
    
    return final_score, reason

def merge_tracks(roi_dir, source_id, dest_id):
    """将一个轨迹的图片合并到另一个"""
    source_path = os.path.join(roi_dir, source_id)
    dest_path = os.path.join(roi_dir, dest_id)
    print(f"Merging track '{source_id}' into '{dest_id}'...")
    for fname in os.listdir(source_path):
        shutil.move(os.path.join(source_path, fname), os.path.join(dest_path, fname))
    os.rmdir(source_path)

def run_clustering_spatiotemporal():
    """
    第二阶段：执行时空感知的轨迹聚类。
    """
    TRACKS_HISTORY_PATH='tracks_history.json'

    with open(TRACKS_HISTORY_PATH, 'r') as f:
        tracks_history = json.load(f)

    # --- 1. 配置 ---
    INITIAL_ROI_DIR = "tracked_rois_initial"
    FINAL_ROI_DIR = "tracked_rois_final"
    MERGE_SCORE_THRESHOLD = 0.75 # 【关键参数】现在是基于最终分数的阈值
    MIN_TRACK_LENGTH = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    VIDEO_PATH = "sample.mp4"
    OUTPUT_JSON_PATH = "tracks_history.json"

    if os.path.exists(FINAL_ROI_DIR): shutil.rmtree(FINAL_ROI_DIR)
    shutil.copytree(INITIAL_ROI_DIR, FINAL_ROI_DIR)
    OUTPUT_LOG_PATH = "testing/merge_log.json"

    # --- 2. 加载模型 & 构建初始数据 ---
    extractor = get_feature_extractor(DEVICE)
    preprocessor = get_preprocess_transform()
    
    track_ids = [d for d in os.listdir(FINAL_ROI_DIR) if os.path.isdir(os.path.join(FINAL_ROI_DIR, d))]
    
    galleries = {}
    track_time_intervals = {} 

    for tid in tqdm(track_ids):
        galleries[tid] = build_gallery(FINAL_ROI_DIR, tid, extractor, preprocessor, DEVICE)
        track_time_intervals[tid] = get_track_time_intervals(FINAL_ROI_DIR, tid)
    

    # 移除无效轨迹
    valid_ids = [tid for tid in track_ids if galleries.get(tid) is not None and track_time_intervals.get(tid)]
    galleries = {tid: galleries[tid] for tid in valid_ids}
    track_time_intervals = {tid: track_time_intervals[tid] for tid in valid_ids}
    merge_log = {tid: {tid} for tid in galleries.keys()}
    
    # --- 3. 迭代合并 ---
    print("\nStarting iterative merging with spatiotemporal scoring...")
    while True: # 一直循环，直到某一轮没有任何合并发生
        merged_in_this_pass = False
        
        # 在每一轮循环开始时，都获取最新的 track_id 列表
        current_track_ids = sorted(list(galleries.keys()))

        ids_to_delete = set()

        for i in range(len(current_track_ids)):
            for j in range(i + 1, len(current_track_ids)):
                id1 = current_track_ids[i]
                id2 = current_track_ids[j]

                # 如果其中一个ID已经被标记为删除，就跳过
                if id1 in ids_to_delete or id2 in ids_to_delete:
                    continue

                score, reason = calculate_spatiotemporal_score(
                    FINAL_ROI_DIR, id1, id2,
                    galleries[id1], galleries[id2],
                    track_time_intervals[id1], track_time_intervals[id2],
                    tracks_history,
                    merge_log,
                    extractor, preprocessor, DEVICE
                )
                print(f"Comparing {id1} and {id2}: Score = {score:.3f} (Reason: {reason})")

                if score > MERGE_SCORE_THRESHOLD:
                    # 合并逻辑保持不变，但更新数据结构时有变化
                    dest_id, source_id = (id1, id2) if track_time_intervals[id1][0][0] < track_time_intervals[id2][0][0] else (id2, id1)
                    
                    merge_tracks(FINAL_ROI_DIR, source_id, dest_id)

                    merge_log[dest_id].update(merge_log[source_id])
                    del merge_log[source_id] # 从日志中删除旧的key
                    
                    # 【修改】更新时间段列表：直接追加并重新排序
                    track_time_intervals[dest_id].extend(track_time_intervals[source_id])
                    track_time_intervals[dest_id].sort()
                    # 也可以在这里添加一个合并相邻区间的逻辑，但非必须
                    
                    # 更新画廊
                    galleries[dest_id] = build_gallery(FINAL_ROI_DIR, dest_id, extractor, preprocessor, DEVICE)
                    
                    ids_to_delete.add(source_id)
                    merged_in_this_pass = True

        # 在一轮比较结束后，统一删除被标记的ID
        if ids_to_delete:
            for tid in ids_to_delete:
                del galleries[tid]
                del track_time_intervals[tid]
        
        # 如果这一整轮都没有任何合并发生，说明聚类已经稳定，可以结束了
        if not merged_in_this_pass:
            break
    
    # --- 5. 最终清理 ---
    print("\nClustering and merging complete. Cleaning up short tracks...")
    final_track_ids = os.listdir(FINAL_ROI_DIR)
    deleted_count = 0
    for tid in final_track_ids:
        track_path = os.path.join(FINAL_ROI_DIR, tid)
        if len(os.listdir(track_path)) < MIN_TRACK_LENGTH:
            shutil.rmtree(track_path)
            deleted_count += 1
            
    print(f"Deleted {deleted_count} tracks with fewer than {MIN_TRACK_LENGTH} frames.")
    print(f"\nFinal number of unique people identified: {len(os.listdir(FINAL_ROI_DIR))}")

    final_track_ids = os.listdir(FINAL_ROI_DIR)
    print(f"\nFinal clusters are: {final_track_ids}")

    # --- 【新增】保存最终结果到JSON ---
    final_merge_log = {k: sorted(list(v)) for k, v in merge_log.items()}

    print(f"\nSaving merge log to '{OUTPUT_LOG_PATH}'...")
    with open(OUTPUT_LOG_PATH, 'w') as f:
        json.dump(final_merge_log, f, indent=4)
    print("Clustering complete. Final ROIs are in 'tracked_rois_final'.")
    print(f"Merge log created at '{OUTPUT_LOG_PATH}'.")



    create_annotated_video(
        original_video_path=VIDEO_PATH,
        tracks_history_path=OUTPUT_JSON_PATH,
        output_video_path="testing/run_cluster.mp4",
        merge_log=final_merge_log,
        interpolate=False 
    )



if __name__ == "__main__":
    run_clustering_spatiotemporal()