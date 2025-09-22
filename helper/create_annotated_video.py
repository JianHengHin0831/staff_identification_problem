import cv2
import json
import numpy as np
import os
from tqdm import tqdm
import config

def linear_interpolate(p1, p2, t):
    return p1 * (1 - t) + p2 * t

def create_annotated_video(
    output_video_path,
    tracks_history_path=config.HISTORY_JSON_PATH,
    original_video_path=config.VIDEO_PATH,
    merge_log=None,
    highlight_ids=None,
    interpolate=False,
    max_interpolation_gap=15,
    show_coordinates=False,
):
    print(f"\n--- Starting Annotated Video Generation for: {os.path.basename(output_video_path)} ---")

    # 1. validate file
    required_files = [original_video_path, tracks_history_path]
    if merge_log and not isinstance(merge_log, dict): 
        required_files.append(merge_log)
        
    for f_path in required_files:
        if isinstance(f_path, str) and not os.path.exists(f_path):
            print(f"Error: Required file not found: '{f_path}'")
            return
            
    # 2. loaded data
    with open(tracks_history_path, 'r') as f:
        all_tracks_data = json.load(f)
        
    # get merge log
    if merge_log and isinstance(merge_log, str):
        with open(merge_log, 'r') as f:
            merge_log = json.load(f)

    # 3. get final_tracks_to_draw
    final_tracks_to_draw = {}
    id_to_color_map = {}
    
    if merge_log:
        print("Mode: Final Cluster Visualization (using merge_log)")
        for final_id, original_ids in merge_log.items():
            if highlight_ids and final_id not in highlight_ids:
                continue
            
            combined_data = {}
            for original_id in original_ids:
                if original_id in all_tracks_data:
                    combined_data.update(all_tracks_data[original_id])
            
            if combined_data:
                final_tracks_to_draw[final_id] = {int(k): v for k, v in combined_data.items()}

    else:
        print("Mode: Initial Tracker Debug (no merge_log)")
        for original_id, track_data in all_tracks_data.items():
            if highlight_ids and original_id not in highlight_ids:
                continue
            final_tracks_to_draw[original_id] = {int(k): v for k, v in track_data.items()}
            
    # 4. distribute color for difference id
    for display_id in final_tracks_to_draw.keys():
        np.random.seed(hash(display_id) & 0xFFFFFFFF)
        id_to_color_map[display_id] = tuple(np.random.randint(50, 256, 3).tolist())
        
    # 5. initialize video writer
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{original_video_path}'")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 6. rendering and plotting
    with tqdm(total=total_frames, desc=f"Rendering {os.path.basename(output_video_path)}") as pbar:
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            for display_id, track_data in final_tracks_to_draw.items():
                bbox = None
                is_interpolated = False
                color = id_to_color_map[display_id]
                
                if frame_idx in track_data:
                    bbox = track_data[frame_idx]
                elif interpolate:
                    sorted_frames = sorted(track_data.keys())
                    
                    prev_frame = -1
                    for f in sorted_frames:
                        if f < frame_idx: prev_frame = f
                        else: break

                    next_frame = -1
                    for f in sorted_frames:
                        if f > frame_idx:
                            next_frame = f
                            break
                    
                    if prev_frame != -1 and next_frame != -1 and (next_frame - prev_frame) <= max_interpolation_gap:
                        t = (frame_idx - prev_frame) / (next_frame - prev_frame)
                        p1_start = np.array(track_data[prev_frame][:2]); p1_end = np.array(track_data[prev_frame][2:])
                        p2_start = np.array(track_data[next_frame][:2]); p2_end = np.array(track_data[next_frame][2:])
                        interp_start = linear_interpolate(p1_start, p2_start, t).astype(int)
                        interp_end = linear_interpolate(p1_end, p2_end, t).astype(int)
                        bbox = [*interp_start, *interp_end]
                        is_interpolated = True
                
                if bbox:
                    x1, y1, x2, y2 = map(int, bbox)
                    final_color = (0, 255, 255) if is_interpolated else color
                    text = display_id.replace("track_", "ID ")
                    if is_interpolated: text += " (interp.)"                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), final_color, 2)                    
                    if show_coordinates: 
                        center_x = (x1+x2) // 2
                        center_y = (y1+y2) // 2
                        coord_Text = "( "+str(center_x) + " , "+str(center_y)+" )"
                        cv2.putText(frame, text, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, final_color, 2)
                        cv2.putText(frame, coord_Text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, final_color, 2)
                    else:
                        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, final_color, 2)

            out.write(frame)
            pbar.update(1)

    # --- 7. release sources ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Annotated video saved to: '{output_video_path}'")