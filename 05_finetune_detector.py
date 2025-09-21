import os
from ultralytics import YOLO
import yaml
import config

def create_yolo_config(dataset_base_path):
    """
    自动创建YOLOv8训练所需的data.yaml配置文件。
    """
    config = {
        'path': os.path.abspath(dataset_base_path),  # 数据集根目录的绝对路径
        'train': os.path.join('images', 'train'),    # train images (相对于 'path')
        'val': os.path.join('images', 'train'),      # val images (我们复用训练集做验证)
        'nc': 1,                                     # 类别数量
        'names': ['staff_tag']                       # 类别名称列表
    }
    
    config_path = os.path.join(dataset_base_path, 'data.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"YOLO config file created at: {config_path}")
    return config_path
 
def train_detector():
    """
    使用合成数据集微调一个YOLOv8n模型来检测员工名牌。
    """
    # --- 1. 配置 ---
    MODEL_TO_FINETUNE = 'yolov8l.pt' 
    EPOCHS = 50
    IMG_SIZE = 320 
    BATCH_SIZE = 16
    PROJECT_NAME = "staff_tag_detector"
    
    # --- 2. 准备配置文件 ---
    if not os.path.exists(config.FINETUNE_DATASET):
        print(f"Error: Dataset directory not found at '{config.FINETUNE_DATASET}'")
        print("Please run the 'generate_finetune_dataset_yolo.py' script first.")
        return
        
    config_path = create_yolo_config(config.FINETUNE_DATASET)
    
    # --- 3. 加载模型并开始训练 ---
    print(f"Loading pre-trained model: {MODEL_TO_FINETUNE}")
    model = YOLO(MODEL_TO_FINETUNE)
    
    print("Starting model fine-tuning...")
    # 使用 .train() 方法进行训练
    results = model.train(
        data=config_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=PROJECT_NAME, # 结果将保存在 runs/detect/staff_tag_detector
        project="runs/detect",
        exist_ok=True # 如果文件夹已存在，则覆盖
    )
    
    print("\nFine-tuning complete!")
    print(f"The best model is saved at: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    train_detector()