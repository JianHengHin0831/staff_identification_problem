# generate_finetune_dataset_yolo.py

import os
import cv2
import numpy as np
import shutil
import random
from tqdm import tqdm
import config

def augment_and_resize_tag(tag_rgba_image, target_width):
    """
    对RGBA名牌进行虚拟俯视变换和增强，并缩放到目标宽度。
    返回一个清理过的BGR图像和一个最终的Alpha Mask。
    """
    if tag_rgba_image is None or tag_rgba_image.shape[2] != 4:
        raise ValueError("Input image must be a 4-channel RGBA image.")
    
    # 分离通道
    bgr_original = tag_rgba_image[:, :, :3]
    alpha_original = tag_rgba_image[:, :, 3]
    
    h, w = bgr_original.shape[:2]
    
    # --- 1. 几何变换 (同时作用于BGR和Alpha) ---
    angle = np.random.uniform(-30, 30)
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated_bgr = cv2.warpAffine(bgr_original, M_rot, (w, h), borderValue=(0,0,0))
    rotated_alpha = cv2.warpAffine(alpha_original, M_rot, (w, h), borderValue=(0))

    shift_x = np.random.uniform(0.05, 0.15) * w
    shift_y = np.random.uniform(0.1, 0.25) * h
    src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dst_pts = np.float32([[0, 0], [w - 1, 0], [shift_x, h - 1 - shift_y], [w - 1 - shift_x, h - 1 - shift_y]])
    M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    final_h, final_w = int(h - shift_y), w
    
    topview_bgr = cv2.warpPerspective(rotated_bgr, M_persp, (final_w, final_h))
    topview_alpha = cv2.warpPerspective(rotated_alpha, M_persp, (final_w, final_h))

    # --- 2. 清理变换引入的边界像素 ---
    cleaned_bgr = cv2.bitwise_and(topview_bgr, topview_bgr, mask=topview_alpha)
    
    # --- 3. 颜色增强 (只作用于清理后的BGR) ---
    alpha_jitter = np.random.uniform(0.8, 1.2)
    beta_jitter = np.random.uniform(-20, 20)
    adjusted_bgr = np.clip(alpha_jitter * cleaned_bgr + beta_jitter, 0, 255).astype(np.uint8)

    # --- 4. 最终缩放 ---
    aspect_ratio = final_w / max(1, final_h)
    target_height = int(target_width / aspect_ratio)
    
    final_bgr = cv2.resize(adjusted_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)
    final_alpha = cv2.resize(topview_alpha, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    
    return final_bgr, final_alpha

def run_synthesis_final_balanced():
    # --- 1. 配置 ---
    
    NUM_POSITIVE_SAMPLES_PER_BG = 3
    NUM_HARD_NEGATIVE_SAMPLES_PER_BG = 2
    
    # --- 2. 准备目录 ---
    print("Preparing dataset directories...")
    images_dir = os.path.join(config.FINETUNE_DATASET, 'images', 'train')
    labels_dir = os.path.join(config.FINETUNE_DATASET, 'labels', 'train')
    if os.path.exists(config.FINETUNE_DATASET): shutil.rmtree(config.FINETUNE_DATASET)
    os.makedirs(images_dir)
    os.makedirs(labels_dir)

    # --- 3. 加载模板和文件列表 ---
    tag_template_rgba = cv2.imread(config.STAFF_TAG_TEMPLATE_PATH, cv2.IMREAD_UNCHANGED)
    distractor_template_rgba = cv2.imread(config.DISTRACTOR_TEMPLATE_PATH, cv2.IMREAD_UNCHANGED)
    if tag_template_rgba is None or distractor_template_rgba is None or \
       tag_template_rgba.shape[2] != 4 or distractor_template_rgba.shape[2] != 4:
        print("Error: Could not load templates as RGBA. Ensure they are valid PNGs with transparency.")
        return
        
    background_files = [f for f in os.listdir(config.EXTERNAL_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not background_files:
        print(f"Error: No background images found in '{config.EXTERNAL_IMAGES_DIR}'")
        return

    print(f"Found {len(background_files)} background images.")
    
    # --- 4. 开始合成 ---
    for bg_fname in tqdm(background_files, desc="Synthesizing balanced dataset"):
        bg_path = os.path.join(config.EXTERNAL_IMAGES_DIR, bg_fname)
        background_img = cv2.imread(bg_path)
        if background_img is None: continue
        bh, bw = background_img.shape[:2]

        # a) 生成正样本 (带标签)
        for i in range(NUM_POSITIVE_SAMPLES_PER_BG):
            target_tag_width = random.randint(bw // 8, bw // 4)
            augmented_tag_bgr, augmented_alpha_mask = augment_and_resize_tag(tag_template_rgba, target_tag_width)
            th, tw = augmented_tag_bgr.shape[:2]
            
            paste_x_min, paste_x_max = int(bw * 0.3), int(bw * 0.7) - tw
            paste_y_min, paste_y_max = int(bh * 0.2), int(bh * 0.5) - th
            if paste_x_max <= paste_x_min or paste_y_max <= paste_y_min: continue
                
            px, py = random.randint(paste_x_min, paste_x_max), random.randint(paste_y_min, paste_y_max)
            
            synthetic_img = background_img.copy()
            pasted_region = synthetic_img[py:py+th, px:px+tw]
            mask_blurred = cv2.GaussianBlur(augmented_alpha_mask, (5, 5), 0)
            mask_float = mask_blurred.astype(float) / 255.0
            mask_3d = np.stack([mask_float]*3, axis=2)
            blended_region = (augmented_tag_bgr * mask_3d) + (pasted_region * (1 - mask_3d))
            synthetic_img[py:py+th, px:px+tw] = blended_region.astype('uint8')
            
            image_save_name = f"{os.path.splitext(bg_fname)[0]}_positive_{i}.jpg"
            image_save_path = os.path.join(images_dir, image_save_name)
            cv2.imwrite(image_save_path, synthetic_img)
            
            x_center, y_center = (px + tw / 2) / bw, (py + th / 2) / bh
            width_norm, height_norm = tw / bw, th / bh
            label_fname = f"{os.path.splitext(bg_fname)[0]}_positive_{i}.txt"
            label_path = os.path.join(labels_dir, label_fname)
            with open(label_path, 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")

        # b) 生成困难负样本 (Hard Negatives) - (不带标签)
        for i in range(NUM_HARD_NEGATIVE_SAMPLES_PER_BG):
            target_tag_width = random.randint(bw // 8, bw // 4)
            augmented_distractor_bgr, augmented_distractor_mask = augment_and_resize_tag(distractor_template_rgba, target_tag_width)
            th, tw = augmented_distractor_bgr.shape[:2]

            paste_x_min, paste_x_max = int(bw * 0.3), int(bw * 0.7) - tw
            paste_y_min, paste_y_max = int(bh * 0.2), int(bh * 0.5) - th
            if paste_x_max <= paste_x_min or paste_y_max <= paste_y_min: continue
                
            px, py = random.randint(paste_x_min, paste_x_max), random.randint(paste_y_min, paste_y_max)
            
            negative_img = background_img.copy()
            pasted_region_neg = negative_img[py:py+th, px:px+tw]
            mask_blurred_neg = cv2.GaussianBlur(augmented_distractor_mask, (5, 5), 0)
            mask_float_neg = mask_blurred_neg.astype(float) / 255.0
            mask_3d_neg = np.stack([mask_float_neg]*3, axis=2)
            blended_region_neg = (augmented_distractor_bgr * mask_3d_neg) + (pasted_region_neg * (1 - mask_3d_neg))
            negative_img[py:py+th, px:px+tw] = blended_region_neg.astype('uint8')
            
            save_path_neg_hard = os.path.join(images_dir, f"{os.path.splitext(bg_fname)[0]}_hard_neg_{i}.jpg")
            cv2.imwrite(save_path_neg_hard, negative_img)
            
        # c) 生成简单负样本 (Easy Negatives) - (不带标签)
        save_path_neg_easy = os.path.join(images_dir, bg_fname)
        shutil.copy(bg_path, save_path_neg_easy)
        
    print("\nBalanced synthetic dataset for YOLO generation complete!")

if __name__ == "__main__":
    run_synthesis_final_balanced()