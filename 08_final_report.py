import json
import os
import config 
from helper.create_annotated_video import create_annotated_video

def frames_to_intervals(frame_list, max_gap=1):
    if not frame_list:
        return []
    
    intervals = []
    start_of_interval = frame_list[0]
    for i in range(1, len(frame_list)):
        if frame_list[i] - frame_list[i-1] > max_gap:
            intervals.append([start_of_interval, frame_list[i-1]])
            start_of_interval = frame_list[i]
    intervals.append([start_of_interval, frame_list[-1]])
    return intervals

def create_report():
    # validate data
    if not os.path.exists(config.STAFF_ID_PATH):
        print(f"Error: Final staff history file not found at '{config.STAFF_ID_PATH}'")
        return
        
    with open(config.STAFF_ID_PATH, 'r') as f:
        staff_data = json.load(f)

    # generate report content
    
    report_content = "AI Evaluation Task Results\n"
    report_content += "="*28 + "\n\n"

    if not staff_data:
        report_content += "No staff members were identified in the video clip.\n"
    else:
        num_staff = len(staff_data)
        report_content += f"A total of {num_staff} staff member(s) were identified.\n"
        report_content += "Details for each staff member are listed below:\n\n"

        for i, (staff_id, track_data) in enumerate(staff_data.items()):
            report_content += f"--- Staff Member #{i+1} (Internal ID: {staff_id}) ---\n\n"

            sorted_frames = sorted([int(k) for k in track_data.keys()])

            # --- Task 1: Identify which frames in the clip have the staff present? ---
            report_content += "Task 1: Presence by Frame Intervals\n"
            report_content += "-------------------------------------\n"
            
            presence_intervals = frames_to_intervals(sorted_frames)
            report_content += f"This staff member was present in the following frame intervals:\n"
            if not presence_intervals:
                report_content += "  - No presence detected.\n"
            for interval in presence_intervals:
                report_content += f"  - Frames: {interval[0]} to {interval[1]}\n"
            report_content += "\n"

            # --- Task 2: Locate the staff xy coordinates when present in the clip. ---
            report_content += "Task 2: XY Coordinates per Frame (Sample)\n"
            report_content += "-----------------------------------------\n"
            report_content += "A sample of the center (x, y) coordinates is shown below.\n"
            
            for frame_idx in sorted_frames[:5]:
                bbox = track_data.get(str(frame_idx))
                if bbox:
                    center_x = (bbox[0] + bbox[2]) // 2
                    center_y = (bbox[1] + bbox[3]) // 2
                    report_content += f"  - Frame {frame_idx}: ({center_x}, {center_y})\n"
            
            if len(sorted_frames) > 5:
                report_content += "  - ... (and more)\n"
            
            report_content += "\n"

    # generate video and report
    report_content += "--- Additional Information ---\n"
    report_content += f"Complete XY coordinates for all identified staff are located in the JSON file: '{os.path.basename(config.STAFF_ID_PATH)}'\n"

    print("Generating final annotated video for all identified staff...")
    create_annotated_video(
        tracks_history_path=config.STAFF_ID_PATH,
        output_video_path=config.REPORT_VID,
        show_coordinates=True,
    )
    
    report_content += f"A complete visualization of all identified staff tracks is available in the video file: '{os.path.basename(config.REPORT_VID)}'\n"

    with open(config.REPORT_TEXT, 'w') as f:
        f.write(report_content)

    print(f"\nReport successfully generated at: '{config.REPORT_TEXT}'")

if __name__ == "__main__":
    create_report()