# run_cluster.py

import os
import shutil
import torch
from tqdm import tqdm
import json
from helper.create_annotated_video import create_annotated_video

# 从共享文件中导入辅助函数
from helper.shared_utils import get_feature_extractor, get_preprocess_transform
import config
from helper.cluster import merge_tracks, build_gallery, get_track_time_intervals, calculate_spatiotemporal_score

def run_clustering_spatiotemporal():
    """
    第二阶段：执行时空感知的轨迹聚类。
    """

    with open(config.HISTORY_JSON_PATH, 'r') as f:
        tracks_history = json.load(f)

    # --- 1. 配置 ---
    MERGE_SCORE_THRESHOLD = 0.75 # 【关键参数】现在是基于最终分数的阈值
    MIN_TRACK_LENGTH = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.exists(config.MERGED_ROI_DIR): shutil.rmtree(config.MERGED_ROI_DIR)
    shutil.copytree(config.TRACKER_ROI_DIR, config.MERGED_ROI_DIR)

    # --- 2. 加载模型 & 构建初始数据 ---
    extractor = get_feature_extractor(DEVICE)
    preprocessor = get_preprocess_transform()
    
    track_ids = [d for d in os.listdir(config.MERGED_ROI_DIR) if os.path.isdir(os.path.join(config.MERGED_ROI_DIR, d))]
    
    galleries = {}
    track_time_intervals = {} 

    for tid in tqdm(track_ids):
        galleries[tid] = build_gallery(config.MERGED_ROI_DIR, tid, extractor, preprocessor, DEVICE)
        track_time_intervals[tid] = get_track_time_intervals(config.MERGED_ROI_DIR, tid)
    

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
                    config.MERGED_ROI_DIR, id1, id2,
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
                    
                    merge_tracks(config.MERGED_ROI_DIR, source_id, dest_id)

                    merge_log[dest_id].update(merge_log[source_id])
                    del merge_log[source_id] # 从日志中删除旧的key
                    
                    # 【修改】更新时间段列表：直接追加并重新排序
                    track_time_intervals[dest_id].extend(track_time_intervals[source_id])
                    track_time_intervals[dest_id].sort()
                    # 也可以在这里添加一个合并相邻区间的逻辑，但非必须
                    
                    # 更新画廊
                    galleries[dest_id] = build_gallery(config.MERGED_ROI_DIR, dest_id, extractor, preprocessor, DEVICE)
                    
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
    final_track_ids = os.listdir(config.MERGED_ROI_DIR)
    deleted_count = 0
    for tid in final_track_ids:
        track_path = os.path.join(config.MERGED_ROI_DIR, tid)
        if len(os.listdir(track_path)) < MIN_TRACK_LENGTH:
            shutil.rmtree(track_path)
            deleted_count += 1
            
    print(f"Deleted {deleted_count} tracks with fewer than {MIN_TRACK_LENGTH} frames.")
    print(f"\nFinal number of unique people identified: {len(os.listdir(config.MERGED_ROI_DIR))}")

    final_track_ids = os.listdir(config.MERGED_ROI_DIR)
    print(f"\nFinal clusters are: {final_track_ids}")

    # --- 【新增】保存最终结果到JSON ---
    final_merge_log = {k: sorted(list(v)) for k, v in merge_log.items()}

    print(f"\nSaving merge log to '{config.MERGED_ROI_LOG}'...")
    with open(config.MERGED_ROI_LOG, 'w') as f:
        json.dump(final_merge_log, f, indent=4)
    print("Clustering complete. Final ROIs are in 'tracked_rois_final'.")
    print(f"Merge log created at '{config.MERGED_ROI_LOG}'.")



    create_annotated_video(
        output_video_path=config.CLUSTER_VID,
        merge_log=final_merge_log,
        interpolate=False 
    )



if __name__ == "__main__":
    run_clustering_spatiotemporal()