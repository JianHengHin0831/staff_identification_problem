# apply_manual_merge.py

import os
import shutil
import json
from helper.create_annotated_video import create_annotated_video
import config

def merge_folders(source_folder, dest_folder):
    """将一个文件夹的内容移动到另一个文件夹，然后删除源文件夹"""
    if not os.path.exists(dest_folder) or not os.path.exists(source_folder):
        print(f"Warning: Skipping merge. Folder not found. Source: {source_folder}, Dest: {dest_folder}")
        return
    for filename in os.listdir(source_folder):
        shutil.move(os.path.join(source_folder, filename), os.path.join(dest_folder, filename))
    os.rmdir(source_folder)
    print(f"  - Merged '{os.path.basename(source_folder)}' into '{os.path.basename(dest_folder)}'.")

def run_apply_manual_merge():
    """
    读取配置文件，执行手动合并，并生成所有最终产物。
    """
    # --- 2. 验证和准备数据 ---
    required = [config.MERGED_ROI_DIR, config.MERGED_ROI_LOG]
    if not all(os.path.exists(f) for f in required):
        print("Error: One or more required input files are missing.")
        print(f"Please ensure these exist: {required}")
        return

    # 创建一个干净的工作副本
    if os.path.exists(config.FINAL_ROI_DIR): shutil.rmtree(config.FINAL_ROI_DIR)
    shutil.copytree(config.MERGED_ROI_DIR, config.FINAL_ROI_DIR)
    
    with open( config.MERGED_ROI_LOG, 'r') as f:
        merge_log = json.load(f)
    
    manual_rules = {                
        "track_00363": ["track_00827","track_00858", "track_01176"]      
    }
    
    print("\n" + "="*50)
    print("        Applying Manual Merge Rules")
    print("="*50)

    # --- 3. 执行合并 ---
    for dest_id, source_ids in manual_rules.items():
        print(f"\nProcessing rule: Merge into '{dest_id}'")
        for source_id in source_ids:
            # a) 合并文件夹
            source_folder_path = os.path.join(config.FINAL_ROI_DIR, source_id)
            dest_folder_path = os.path.join(config.FINAL_ROI_DIR, dest_id)
            merge_folders(source_folder_path, dest_folder_path)
            
            # b) 更新合并日志
            if dest_id in merge_log and source_id in merge_log:
                merge_log[dest_id].extend(merge_log[source_id])
                merge_log[dest_id] = sorted(list(set(merge_log[dest_id])))
                del merge_log[source_id]

    print("\nManual merge operations complete.")

    # --- 4. 保存最终的合并日志 ---
    print(f"\nSaving updated merge log to '{config.FINAL_ROI_LOG}'...")
    with open(config.FINAL_ROI_LOG, 'w') as f:
        json.dump(merge_log, f, indent=4)
    print("Save complete.")

    # --- 5. 生成最终的可视化视频 ---
    print("\nGenerating the final annotated video based on manual merges...")
    
    create_annotated_video(
        output_video_path=config.MANUAL_VID,
        merge_log=merge_log,
        interpolate=False 
    )

if __name__ == "__main__":
    run_apply_manual_merge()