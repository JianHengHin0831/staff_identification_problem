import numpy as np
from helper.create_annotated_video import linear_interpolate


# 2 steps done in this function
#1. Clean the tracks to remove short segments with sudden position changes.
#2. Perform linear interpolation on short missing frames in the cleaned tracks.

def run_clean_and_interpolate(staff_full_data):
    MAX_JUMP_DISTANCE = 200 # the distance that consider as "sudden position changes"
    MAX_FRAGMENT_LENGTH = 60 # remove the frames if sudden change and the change < MAX_FRAGMENT_LENGTH
    MAX_INTERPOLATION_GAP = 30 # only do interpolation that lower the 1s

    # 1. Clean the tracks to remove short segments with sudden position changes.
    cleaned_staff_data = {}
    for staff_id, track_data in staff_full_data.items():
        print(f"  Cleaning track for staff ID: {staff_id}")
        
        sorted_frames = sorted(track_data.keys())
        
        # Split the track into multiple segments based on the "sudden changed" point
        fragments = []
        current_fragment = {}
        
        if sorted_frames:
            current_fragment[sorted_frames[0]] = track_data[sorted_frames[0]]

        
        for i in range(1, len(sorted_frames)):
            prev_frame = sorted_frames[i-1]
            current_frame = sorted_frames[i]
            
            # only check the consecutive frames
            if current_frame - prev_frame ==1:
                prev_bbox = track_data[prev_frame]
                current_bbox = track_data[current_frame]
                prev_center = np.array([(prev_bbox[0]+prev_bbox[2])/2, (prev_bbox[1]+prev_bbox[3])/2])
                current_center = np.array([(current_bbox[0]+current_bbox[2])/2, (current_bbox[1]+current_bbox[3])/2])
                dist = np.linalg.norm(current_center - prev_center)
                
                if dist > MAX_JUMP_DISTANCE:
                    # found "teleport", ended the current segment and started a new segment
                    fragments.append(current_fragment)
                   
                    current_fragment = {}
            
            current_fragment[current_frame] = track_data[current_frame]

        # add last fragment
        fragments.append(current_fragment)
        
        # filter shorter fragments
        final_track_data = {}
        for frag in fragments:
            if len(frag) > MAX_FRAGMENT_LENGTH:
                final_track_data.update(frag)
            else:
                print(f"    - Removed a fragment of length {len(frag)} due to a large jump.")
                
        cleaned_staff_data[staff_id] = final_track_data

    # 2. Perform linear interpolation on short missing frames in the cleaned tracks.
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

    return interpolated_staff_data

  
