
from torchvision import models, transforms
import torch
import cv2
import json
import numpy as np
import os
from tqdm import tqdm

def get_feature_extractor(device):
    """加载预训练的ResNet50作为特征提取器"""
    print("Loading pre-trained ResNet50 model...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model = model.eval().to(device)
    print("Model loaded successfully.")
    return model

def get_preprocess_transform():
    """获取图像预处理的变换"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_features(extractor, preprocessor, image_bgr, device):
    """从BGR格式的图像裁剪中提取特征向量"""
    if image_bgr is None or image_bgr.size == 0:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    preprocessed_img = preprocessor(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        features = extractor(preprocessed_img)
    return torch.nn.functional.normalize(features, p=2, dim=1)

def create_debug_video(original_video_path, tracks_history_path, output_video_path="debug_tracker_output.mp4", merge_log=None):
    """
    创建一个调试视频，将所有被追踪到的轨迹都用不同颜色框起来。
    """
    print("\n--- Starting Debug Video Generation ---")
    
    # --- 1. 验证输入文件 ---
    if not os.path.exists(original_video_path) or not os.path.exists(tracks_history_path):
        print(f"Error: Missing required files for debug video generation.")
        print(f"Check for: '{original_video_path}' and '{tracks_history_path}'")
        return

    # --- 2. 加载数据 ---
    print(f"Loading track history from '{tracks_history_path}'...")
    with open(tracks_history_path, 'r') as f:
        all_tracks_data = json.load(f)

    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{original_video_path}'")
        return

    # --- 3. 视频读写器初始化 ---
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # --- 4. 颜色生成 ---
    # 为每个 track_id 生成一个独特的、固定的颜色
    colors = {}
    if merge_log:
        print(merge_log)
        # a) 如果有合并日志，基于最终的聚类ID生成颜色
        final_cluster_ids = list(merge_log.keys())
        cluster_colors = {}
        for final_id in final_cluster_ids:
            # 用最终ID生成固定颜色
            np.random.seed(hash(final_id) & 0xFFFFFFFF)
            cluster_colors[final_id] = tuple(np.random.randint(50, 256, 3).tolist())
        
        # b) 创建一个从“原始ID”到“最终颜色”的映射 

        for final_id, original_ids in merge_log.items():
            for original_id in original_ids:
                colors[original_id] = cluster_colors[final_id]
      
    else:
        print("No merge log provided. Assigning unique colors to each initial track.")
        # 如果没有合并日志（比如在run_tracker.py中调用），则使用旧逻辑
        track_ids = list(all_tracks_data.keys())
        for tid in track_ids:
            np.random.seed(hash(tid) & 0xFFFFFFFF)
            colors[tid] = tuple(np.random.randint(50, 256, 3).tolist())
            #   except ValueError:
            #     # 如果ID不是数字，使用哈希
            #     np.random.seed(hash(tid) & 0xFFFFFFFF)
            #     colors[tid] = tuple(np.random.randint(50, 256, 3).tolist())

    # --- 5. 逐帧处理和绘制 ---
    print(f"Processing frames to create debug video: {output_video_path}")
    frame_idx = 0
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Creating debug video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break

            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 遍历的是原始的、未聚类的轨迹数据
            for original_track_id, track_data in all_tracks_data.items():
                if str(frame_idx) in track_data:
                    bbox = track_data[str(frame_idx)]
                    x1, y1, x2, y2 = bbox
                    
                    # 使用我们创建的颜色映射来获取颜色
                    
                    color = colors.get(original_track_id, (255, 0, 0)) # 默认为红色
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    display_id = original_track_id.replace("track_", "ID ")
                    cv2.putText(frame, display_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            out.write(frame)
            frame_idx += 1
            pbar.update(1)

    # --- 6. 释放资源 ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nDebug video generation complete!")
    print(f"Output saved to: '{output_video_path}'")