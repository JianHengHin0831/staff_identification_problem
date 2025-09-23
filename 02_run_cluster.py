# run_cluster.py

import os
import shutil
import torch
from tqdm import tqdm
import json
from helper.create_annotated_video import create_annotated_video
from helper.shared_utils import get_feature_extractor, get_preprocess_transform
import config
from helper.cluster import merge_tracks, build_gallery, get_track_time_intervals, calculate_spatiotemporal_score

def run_clustering():
    with open(config.HISTORY_JSON_PATH, 'r') as f:
        tracks_history = json.load(f)

    MERGE_SCORE_THRESHOLD = 0.75 #  > MERGE_SCORE_THRESHOLD, two tracks merged
    MIN_TRACK_LENGTH = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.exists(config.MERGED_ROI_DIR): shutil.rmtree(config.MERGED_ROI_DIR)
    shutil.copytree(config.TRACKER_ROI_DIR, config.MERGED_ROI_DIR)

    # load model
    extractor = get_feature_extractor(DEVICE)
    preprocessor = get_preprocess_transform()
    
    track_ids = [d for d in os.listdir(config.MERGED_ROI_DIR) if os.path.isdir(os.path.join(config.MERGED_ROI_DIR, d))]
    
    galleries = {}
    track_time_intervals = {} 

    for tid in tqdm(track_ids):
        galleries[tid] = build_gallery(config.MERGED_ROI_DIR, tid, extractor, preprocessor, DEVICE)
        track_time_intervals[tid] = get_track_time_intervals(config.MERGED_ROI_DIR, tid)
    
    # remove unused tracks
    valid_ids = [tid for tid in track_ids if galleries.get(tid) is not None and track_time_intervals.get(tid)]
    galleries = {tid: galleries[tid] for tid in valid_ids}
    track_time_intervals = {tid: track_time_intervals[tid] for tid in valid_ids}
    merge_log = {tid: {tid} for tid in galleries.keys()}
    
    print("\nStarting iterative merging with spatiotemporal scoring...")
    # main loop, merging
    while True: # loop until no merging anymore
        merged_in_this_pass = False
        
         # get latest track id list
        current_track_ids = sorted(list(galleries.keys()))

        ids_to_delete = set()

        for i in range(len(current_track_ids)):
            for j in range(i + 1, len(current_track_ids)):
                id1 = current_track_ids[i]
                id2 = current_track_ids[j]

                # if id is deleted, skipped
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
                    # merge logic unchanged
                    dest_id, source_id = (id1, id2) if track_time_intervals[id1][0][0] < track_time_intervals[id2][0][0] else (id2, id1)
                    
                    merge_tracks(config.MERGED_ROI_DIR, source_id, dest_id)

                    merge_log[dest_id].update(merge_log[source_id])
                    del merge_log[source_id] # remove key from merge log
                    
                    # Udirectly append and reorder in track_time_intervals
                    track_time_intervals[dest_id].extend(track_time_intervals[source_id])
                    track_time_intervals[dest_id].sort()
                    
                    # update galleries
                    galleries[dest_id] = build_gallery(config.MERGED_ROI_DIR, dest_id, extractor, preprocessor, DEVICE)
                    
                    ids_to_delete.add(source_id)
                    merged_in_this_pass = True

        # deleted merged ids
        if ids_to_delete:
            for tid in ids_to_delete:
                del galleries[tid]
                del track_time_intervals[tid]
        
        if not merged_in_this_pass: # stable
            break
    
    # clean short tracks
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

    # output results
    final_merge_log = {k: sorted(list(v)) for k, v in merge_log.items()}

    print(f"\nSaving merge log to '{config.MERGED_ROI_LOG}'...")
    with open(config.MERGED_ROI_LOG, 'w') as f:
        json.dump(final_merge_log, f, indent=4)
    print("Clustering complete. Final ROIs are in 'tracked_rois_final'.")
    print(f"Merge log created at '{config.MERGED_ROI_LOG}'.")



    create_annotated_video(
        output_video_path=config.CLUSTER_VID,
        merge_log=final_merge_log
    )



if __name__ == "__main__":
    run_clustering()