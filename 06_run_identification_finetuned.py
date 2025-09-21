import os
import json
from ultralytics import YOLO
from tqdm import tqdm
import torch 
import cv2 
import config

def run_identification_with_custom_detector():
    # --- 1. 配置 ---
    DETECTOR_WEIGHTS_PATH = "runs/detect/staff_tag_detector/weights/best.pt"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    DETECTION_CONFIDENCE = 0.40
    DECISION_THRESHOLD_RATIO = 0.10
    MIN_VOTE_COUNT = 10

    # --- 2. 验证和加载模型 ---
    if not os.path.exists(DETECTOR_WEIGHTS_PATH):
        print(f"Error: Fine-tuned model weights not found at '{DETECTOR_WEIGHTS_PATH}'")
        return
        
    print(f"Loading custom fine-tuned detector from: {DETECTOR_WEIGHTS_PATH}")
    tag_detector = YOLO(DETECTOR_WEIGHTS_PATH)
    tag_detector.to(DEVICE)

    # --- 3. 遍历轨迹文件夹进行投票 ---
    print(f"\nIdentifying staff from tracks in '{config.FINAL_ROI_DIR}'...")
    if not os.path.exists(config.FINAL_ROI_DIR):
        print(f"Error: Directory not found: '{config.FINAL_ROI_DIR}'")
        return
        
    track_ids = [d for d in os.listdir(config.FINAL_ROI_DIR) if os.path.isdir(os.path.join(config.FINAL_ROI_DIR, d))]
    
    staff_identification_results = {}

    for tid in tqdm(track_ids, desc="Processing tracks"):
        track_path = os.path.join(config.FINAL_ROI_DIR, tid)
        
        image_files = []
        for root, _, files in os.walk(track_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        total_images = len(image_files)
        if total_images == 0: continue
        
        vote_count = 0
        
        # 【核心修正】逐一预测，并在调用YOLO前增加一个可靠的图像读取检查
        for img_path in image_files:
            # --- 前置验证 ---
            # 使用 cv2.imread 尝试读取图片
            img = cv2.imread(img_path)
            
            # 如果 img is None，说明图片已损坏或无法读取
            if img is None:
                print(f"\nWarning: Skipping corrupted or unreadable image: {img_path}")
                # 从总数中减去这张坏图片
                total_images -= 1
                continue # 跳到下一张图片
            
            # --- 如果图片有效，才进行预测 ---
            try:
                # 现在可以安全地把图片路径（或已读取的img对象）传给YOLO
                results = tag_detector(img_path, conf=DETECTION_CONFIDENCE, verbose=False)
                
                if len(results[0].boxes) > 0:
                    vote_count += 1
            except Exception as e:
                # 保留一个通用的异常捕捉，以防YOLO内部发生其他错误
                print(f"\nAn unexpected error occurred while processing {img_path}: {e}")
                total_images -= 1

        # --- 4. 最终决策 ---
        vote_ratio = vote_count / total_images if total_images > 0 else 0
        is_staff = vote_count >= MIN_VOTE_COUNT and vote_ratio >= DECISION_THRESHOLD_RATIO
        
        staff_identification_results[tid] = {
            "is_staff": is_staff,
            "vote_count": vote_count,
            "total_images": total_images,
            "vote_ratio": f"{vote_ratio:.2%}"
        }
    # --- 5. 打印并保存最终结果 ---
    print("\n" + "="*50)
    print("      Final Staff Identification Results (Custom Detector)")
    print("="*50)
    
    staff_list = []
    for tid, result in staff_identification_results.items():
        if result['is_staff']:
            staff_list.append(tid)
        print(f"Track ID: {tid} -> Is Staff? {result['is_staff']} "
              f"(Votes: {result['vote_count']}/{result['total_images']}, "
              f"Ratio: {result['vote_ratio']})")
              
    print("\n--- Summary ---")
    print(f"Staff Track IDs: {staff_list if staff_list else 'None'}")
    
    with open(config.STAFF_LIST_PATH, 'w') as f:
        json.dump({"staff_ids": staff_list}, f, indent=4)
    print(f"Staff list saved to '{config.STAFF_LIST_PATH}'")


if __name__ == "__main__":
    run_identification_with_custom_detector()