import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import numpy as np

def stabilize_video(input_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open the video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = os.path.join(os.path.dirname(input_path), "stabilized_output.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    ret, prev_frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Could not read the video file.")
        cap.release()
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    trajectory = np.zeros((1, 2), np.float32)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar['maximum'] = total_frames
    
    frame_count = 0
    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        
        if prev_pts is None:
            prev_gray = curr_gray
            continue
        
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        valid_prev_pts = prev_pts[status == 1]
        valid_curr_pts = curr_pts[status == 1]
        
        if len(valid_prev_pts) < 10:
            prev_gray = curr_gray
            continue
        
        transform_matrix, _ = cv2.estimateAffinePartial2D(valid_prev_pts, valid_curr_pts)
        
        if transform_matrix is None:
            prev_gray = curr_gray
            continue
        
        dx, dy = transform_matrix[0, 2], transform_matrix[1, 2]
        trajectory = np.vstack((trajectory, trajectory[-1] + [dx, dy]))
        smoothed_trajectory = cv2.GaussianBlur(trajectory, (5, 5), 0)
        diff = smoothed_trajectory[-1] - trajectory[-1]
        transform_matrix[0, 2] += diff[0]
        transform_matrix[1, 2] += diff[1]
        
        stabilized_frame = cv2.warpAffine(curr_frame, transform_matrix, (width, height))
        out.write(stabilized_frame)
        prev_gray = curr_gray.copy()
        
        frame_count += 1
        progress_bar['value'] = frame_count
        root.update_idletasks()
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    messagebox.showinfo("Success", f"Video stabilization complete! Saved at: {output_path}")
    os.startfile(output_path) if os.name == 'nt' else os.system(f'open {output_path}')

def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if file_path:
        status_label.config(text=f"Selected: {file_path}")
        stabilize_video(file_path)

# GUI Setup
root = tk.Tk()
root.title("Video Stabilization")
root.geometry("500x250")
root.configure(bg="#2C3E50")

title_label = tk.Label(root, text="Select a Video for Stabilization", font=("Arial", 12, "bold"), fg="white", bg="#2C3E50")
title_label.pack(pady=10)

select_button = tk.Button(root, text="Choose Video", command=select_video, font=("Arial", 10, "bold"), fg="white", bg="#1ABC9C", padx=10, pady=5)
select_button.pack(pady=5)

status_label = tk.Label(root, text="No file selected", font=("Arial", 10), fg="white", bg="#2C3E50")
status_label.pack(pady=5)

progress_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=400, mode='determinate')
progress_bar.pack(pady=10)

root.mainloop()