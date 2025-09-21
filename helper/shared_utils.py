
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

# --- 1. 辅助函数 ---

def calculate_histogram(image_bgr, bins=[8, 8, 8]):
    """计算BGR图像的3D颜色直方图并归一化"""
    if image_bgr is None or image_bgr.size == 0: return None
    hist = cv2.calcHist([image_bgr], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist

def build_reference_data(roi_folder, num_templates=10, resize_dim=(64, 128)):
    """从一个ROI文件夹中创建模板库和平均颜色直方图指纹"""
    templates = []
    hists = []
    image_files = []
    for root, _, files in os.walk(roi_folder):
        for file in files:
            if file.endswith('.jpg'):
                image_files.append(os.path.join(root, file))
    
    if not image_files: return [], None

    sample_size = min(len(image_files), num_templates)
    sample_files = np.random.choice(image_files, sample_size, replace=False)

    for img_path in sample_files:
        img = cv2.imread(img_path)
        if img is not None:
            hists.append(calculate_histogram(img))
            template_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            templates.append(cv2.resize(template_gray, resize_dim))
            
    if not hists: return templates, None
    
    valid_hists = [h for h in hists if h is not None]
    if not valid_hists: return templates, None
    
    return templates, np.mean(valid_hists, axis=0)

def find_best_match_in_frame(frame, templates, ref_histogram, prev_bbox, 
                             search_radius=20, match_threshold=0.1, hist_threshold=0.5):
    """带颜色直方图双重验证的、多尺度的模板匹配"""
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    px, py = (prev_bbox[0] + prev_bbox[2]) // 2, (prev_bbox[1] + prev_bbox[3]) // 2
    x_min = max(0, px - search_radius); y_min = max(0, py - search_radius)
    x_max = min(frame.shape[1], px + search_radius); y_max = min(frame.shape[0], py + search_radius)
    
    search_area_gray = frame_gray[y_min:y_max, x_min:x_max]
    # 【KEPT & EXPLAINED】我们保留这个变量，用于后续高效的颜色ROI裁剪
    search_area_color = frame[y_min:y_max, x_min:x_max]

    if search_area_gray.size == 0: return None
    
    best_overall_score = -1
    final_best_bbox = None

    for template in templates:
        for scale in np.linspace(0.8, 1.2, 5):
            resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
            th, tw = resized_template.shape
            
            if th > search_area_gray.shape[0] or tw > search_area_gray.shape[1]: continue

            res = cv2.matchTemplate(search_area_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > match_threshold:
                top_left_local = max_loc
                
                # 【核心修正】现在我们正确地从更小的 `search_area_color` 中裁剪
                candidate_roi_color = search_area_color[
                    top_left_local[1] : top_left_local[1] + th,
                    top_left_local[0] : top_left_local[0] + tw
                ]

                if candidate_roi_color.size > 0:
                    candidate_hist = calculate_histogram(candidate_roi_color)
                    if candidate_hist is None: continue
                    hist_similarity = cv2.compareHist(ref_histogram, candidate_hist, cv2.HISTCMP_CORREL)

                    if hist_similarity >= hist_threshold:
                        if max_val > best_overall_score:
                            best_overall_score = max_val
                            final_best_bbox = [
                                top_left_local[0] + x_min, top_left_local[1] + y_min,
                                top_left_local[0] + x_min + tw, top_left_local[1] + y_min + th
                            ]
    return final_best_bbox
