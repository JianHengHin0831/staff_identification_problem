# apply_manual_merge.py

import os
import shutil
import json
from helper.create_annotated_video import create_annotated_video
import config

def merge_folders(source_folder, dest_folder):
    # move the contents of one folder to another folder and then delete the source folder
    if not os.path.exists(dest_folder) or not os.path.exists(source_folder):
        print(f"Warning: Skipping merge. Folder not found. Source: {source_folder}, Dest: {dest_folder}")
        return
    for filename in os.listdir(source_folder):
        shutil.move(os.path.join(source_folder, filename), os.path.join(dest_folder, filename))
    os.rmdir(source_folder)
    print(f"  - Merged '{os.path.basename(source_folder)}' into '{os.path.basename(dest_folder)}'.")

def run_apply_manual_merge():
    # validate
    required = [config.MERGED_ROI_DIR, config.MERGED_ROI_LOG]
    if not all(os.path.exists(f) for f in required):
        print("Error: One or more required input files are missing.")
        print(f"Please ensure these exist: {required}")
        return

    # create new folder
    if os.path.exists(config.FINAL_ROI_DIR): shutil.rmtree(config.FINAL_ROI_DIR)
    shutil.copytree(config.MERGED_ROI_DIR, config.FINAL_ROI_DIR)
    
    with open( config.MERGED_ROI_LOG, 'r') as f:
        merge_log = json.load(f)
    
    manual_rules = {                
        "track_00363": ["track_00827", "track_01176"]      
    }
    
    print("\n" + "="*50)
    print("        Applying Manual Merge Rules")
    print("="*50)

    # apply merge
    for dest_id, source_ids in manual_rules.items():
        print(f"\nProcessing rule: Merge into '{dest_id}'")
        for source_id in source_ids:
            # merge folders
            source_folder_path = os.path.join(config.FINAL_ROI_DIR, source_id)
            dest_folder_path = os.path.join(config.FINAL_ROI_DIR, dest_id)
            merge_folders(source_folder_path, dest_folder_path)
            
            # update logs
            if dest_id in merge_log and source_id in merge_log:
                merge_log[dest_id].extend(merge_log[source_id])
                merge_log[dest_id] = sorted(list(set(merge_log[dest_id])))
                del merge_log[source_id]

    print("\nManual merge operations complete.")

    # output results
    print(f"\nSaving updated merge log to '{config.FINAL_ROI_LOG}'...")
    with open(config.FINAL_ROI_LOG, 'w') as f:
        json.dump(merge_log, f, indent=4)
    print("Save complete.")

    print("\nGenerating the final annotated video based on manual merges...")
    
    create_annotated_video(
        output_video_path=config.MANUAL_VID,
        merge_log=merge_log,
        interpolate=False 
    )

if __name__ == "__main__":
    run_apply_manual_merge()