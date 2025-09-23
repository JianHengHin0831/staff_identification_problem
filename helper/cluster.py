import os
import shutil
import torch
import numpy as np
import cv2
from helper.shared_utils import extract_features

def find_closest_transition_frames(roi_dir, track_id1, track_id2, intervals1, intervals2):
    min_gap = float('inf')
    best_frame1 = -1
    best_frame2 = -1
    id_of_best_frame1 = None
    id_of_best_frame2 = None

    # find the shortest intervals
    for start1, end1 in intervals1:
        for start2, end2 in intervals2:
            # compare end1 and start2
            gap1 = abs(start2 - end1)
            if gap1 < min_gap:
                min_gap = gap1
                best_frame1, best_frame2 = end1, start2
                id_of_best_frame1, id_of_best_frame2 = track_id1, track_id2

            # compare end2 and start1
            gap2 = abs(start1 - end2)
            if gap2 < min_gap:
                min_gap = gap2
                best_frame1, best_frame2 = end2, start1 
                id_of_best_frame1, id_of_best_frame2 = track_id2, track_id1
                
    if best_frame1 == -1 or best_frame2 == -1:
        return None, None, None, None, float('inf')

    path1 = os.path.join(roi_dir, id_of_best_frame1, f"frame_{str(best_frame1).zfill(5)}.jpg")
    path2 = os.path.join(roi_dir, id_of_best_frame2, f"frame_{str(best_frame2).zfill(5)}.jpg")

    return path1, path2, best_frame1, best_frame2, min_gap

def build_gallery(roi_dir, track_id, extractor, preprocessor, device, sample_size=15):
    # build a feature gallery for each track
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
        # if the interval between the current frame number - previous frame number > max_gap, 
        # it is considered to be a new time period.
        if frame_numbers[i] - frame_numbers[i-1] > max_gap:
            intervals.append((start_interval, frame_numbers[i-1]))
            start_interval = frame_numbers[i]
    # add last intervals
    intervals.append((start_interval, frame_numbers[-1]))
    
    return intervals

def check_intervals_overlap(intervals1, intervals2):
    #check if overlap
    for start1, end1 in intervals1:
        for start2, end2 in intervals2:
            # check if overlap
            if max(start1, start2) <= min(end1, end2):
                return True
    return False

def get_min_time_gap(intervals1, intervals2):
    # Calculate the minimum interval between two intervals
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

    # 1.check time overlap
    if check_intervals_overlap(intervals1, intervals2):
        return 0.0, "Time Overlap"

    # get min time gap
    min_gap = get_min_time_gap(intervals1, intervals2)
    
    # get factor time based on the time gap
    reason_time=''
    if min_gap <= 30:
        factor_time = 1.25 
        reason_time = f"Very Close Gap ({min_gap}f)"
    elif min_gap <= 300:
        factor_time = 1.15 - (min_gap - 30) * (0.25 / 270)
        reason_time = f"Medium Gap ({min_gap}f)"
    else: 
        factor_time = 0.9 
        reason_time = f"Large Gap ({min_gap}f)"

    # 2.1 calculate the global similarities 
    sim_global = min(
        torch.mm(gallery1, gallery2.T).max(dim=1).values.mean().item(),
        torch.mm(gallery1, gallery2.T).max(dim=0).values.mean().item()
    )

    path1, path2, frame1, frame2, min_gap = find_closest_transition_frames(
        roi_dir, track_id1, track_id2, intervals1, intervals2
    )

    # 2.2 calculate the local similarities
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
        sim_transition = 0.0
            
    # calculate weight of global and local appearance based on min gap
    if min_gap <= 45: 
        w_global, w_transition = 0.4, 0.6
    else:
        w_global, w_transition = 0.7, 0.3
        
    sim_appearance = w_global * sim_global + w_transition * sim_transition
    reason_appearance = f"Global Appearance is {sim_global} Local Appearance is {sim_transition}"

     # 3. find spatial vector 
    score_spatial = 0.0
    dist = float('inf')
    bbox1, bbox2 = None, None
    
    # find the cluster id based on the frame path
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
    
    # 4. calculate the final marks
    w_appearance = 0.85
    w_spatial = 0.15
    reason_spatial=f"Spatial:  {score_spatial}"
    
    combined_score = (w_appearance * sim_appearance + w_spatial * score_spatial)
    final_score = combined_score * factor_time
    reason = reason_time+reason_appearance+reason_spatial
    
    return final_score, reason

def merge_tracks(roi_dir, source_id, dest_id):
    # merge tracks
    source_path = os.path.join(roi_dir, source_id)
    dest_path = os.path.join(roi_dir, dest_id)
    print(f"Merging track '{source_id}' into '{dest_id}'...")
    for fname in os.listdir(source_path):
        shutil.move(os.path.join(source_path, fname), os.path.join(dest_path, fname))
    os.rmdir(source_path)