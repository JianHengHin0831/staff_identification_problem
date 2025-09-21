# Dir
DATASET_DIR = "dataset/"
HELPER_DIR="helper/"
OUTPUT_DIR="output/"
TESTING="testing/"
FINAL_OUTPUT="final_output/"

VIDEO_PATH = DATASET_DIR + "sample.mp4"

# tracker config
HISTORY_JSON_PATH = OUTPUT_DIR + "tracks_history.json" 
TRACKER_ROI_DIR = OUTPUT_DIR + "01_tracker_rois"

# cluster config
MERGED_ROI_LOG = OUTPUT_DIR + "merge_log.json"
MERGED_ROI_DIR = OUTPUT_DIR + "02_merge_rois"

# manualy merge dataset
FINAL_ROI_DIR = OUTPUT_DIR +  "03_final_rois"
FINAL_ROI_LOG = OUTPUT_DIR + "merge_log_manual.json"

# dataset used to be finetune staff tag detector
EXTERNAL_IMAGES_DIR = DATASET_DIR + "external_person_images/" #dataset used to be create dataset (P-DESTRE)
FINETUNE_DATASET = OUTPUT_DIR + "finetune_dataset/" #dataset used to be trained (created by external image + template)
STAFF_TAG_TEMPLATE_PATH = DATASET_DIR + "staff_tag/staff_tag_template_1.png" # template staff tag
DISTRACTOR_TEMPLATE_PATH = DATASET_DIR + "staff_tag/distractor_template.png" #noise data

# identification
STAFF_LIST_PATH = OUTPUT_DIR + "staff_list.json"

# staff extend
STAFF_ID_PATH = OUTPUT_DIR +"staff_history.json"

# visualization
TRACKER_VID = TESTING + "01_tracker.mp4"
CLUSTER_VID = TESTING + "02_cluster.mp4"
MANUAL_VID = TESTING + "03_manual.mp4"






