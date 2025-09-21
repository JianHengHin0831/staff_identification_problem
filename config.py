# Dir
DATASET_DIR = "dataset/"
HELPER_DIR="helper/"
OUTPUT_DIR="output/"
TESTING="testing/"
FINAL_OUTPUT="final_output/"

VIDEO_PATH = DATASET_DIR + "sample.mp4"

# tracker config
HISTORY_JSON_PATH = "tracks_history.json" 
TRACKER_ROI_DIR = "tracked_rois_initial"

# cluster config
MERGED_ROI_DIR = "tracked_rois_final"
MERGED_ROI_LOG = OUTPUT_DIR + "merge_log.json"

# manualy merge dataset
FINAL_ROI_DIR = "tracked_rois_manual_final"
FINAL_ROI_LOG = OUTPUT_DIR + "merge_log_manual.json"

# dataset used to be finetune staff tag detector
EXTERNAL_IMAGES_DIR = "external_person_images/" #dataset used to be create dataset (P-DESTRE)
FINETUNE_DATASET = "finetune_dataset/" #dataset used to be trained (created by external image + template)
STAFF_TAG_TEMPLATE_PATH = "staff_tag/staff_tag_template_1.png" # template staff tag
DISTRACTOR_TEMPLATE_PATH = "staff_tag/distractor_template.png" #noise data

# identification
STAFF_LIST_PATH = "staff_list.json"

# staff extend
STAFF_ID_PATH = "staff_history_extended_no_ai.json"

# visualization
TRACKER_VID = TESTING + "01_tracker.mp4"
CLUSTER_VID = TESTING + "02_cluster.mp4"
MANUAL_VID = TESTING + "03_manual.mp4"






