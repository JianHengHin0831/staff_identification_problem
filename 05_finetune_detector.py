import os
from ultralytics import YOLO
import yaml
import config

def create_yolo_config(dataset_base_path):
    config = {
        'path': os.path.abspath(dataset_base_path),  # absolute path for dataset
        'train': os.path.join('images', 'train'),    # train images
        'val': os.path.join('images', 'train'),      # val images 
        'nc': 1,                                     # number of channel/ classifier
        'names': ['staff_tag']                       # class list
    }
    
    config_path = os.path.join(dataset_base_path, 'data.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"YOLO config file created at: {config_path}")
    return config_path
 
def train_detector():
    # detect name tag using yolo model  
    MODEL_TO_FINETUNE = 'yolov8l.pt' 
    EPOCHS = 50
    IMG_SIZE = 320 
    BATCH_SIZE = 16
    PROJECT_NAME = "staff_tag_detector"
    
    # find FINETUNE_DATASET
    if not os.path.exists(config.FINETUNE_DATASET):
        print(f"Error: Dataset directory not found at '{config.FINETUNE_DATASET}'")
        print("Please run the 'generate_finetune_dataset_yolo.py' script first.")
        return
        
    config_path = create_yolo_config(config.FINETUNE_DATASET)
    
    # load model and start training
    print(f"Loading pre-trained model: {MODEL_TO_FINETUNE}")
    model = YOLO(MODEL_TO_FINETUNE)
    
    print("Starting model fine-tuning...")
    results = model.train(
        data=config_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=PROJECT_NAME,
        project="runs/detect",
        exist_ok=True 
    )
    
    print("\nFine-tuning complete!")
    print(f"The best model is saved at: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    train_detector()