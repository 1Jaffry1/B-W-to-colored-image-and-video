import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import time
import app
import os
import numpy as np

class VideoPlayer:
    def __init__(self, label, video_path):
        self.label = label
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.playing = False
        self.current_frame = 0

    def start(self):
        self.playing = True
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        threading.Thread(target=self.update_frame, daemon=True).start()

    def stop(self):
        self.playing = False
        self.current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        
    def update_frame(self):
        while self.playing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (500, 500))
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.label.config(image=img)
            self.label.image = img
            time.sleep(1/self.cap.get(cv2.CAP_PROP_FPS))
        self.playing = False

class ColorizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image/Video Colorization")
        self.root.geometry("1200x800")
        
        # Video player instances
        self.input_player = None
        self.output_player = None
        self.playing = False
        
        # Initialize UI
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left pane (input)
        self.left_frame = ttk.Frame(main_frame, width=550, relief=tk.SUNKEN)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right pane (output)
        self.right_frame = ttk.Frame(main_frame, width=550, relief=tk.SUNKEN)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control buttons
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.open_btn = ttk.Button(control_frame, text="Open File", command=self.open_file)
        self.open_btn.pack(side=tk.LEFT, padx=5)
        
        self.start_btn = ttk.Button(control_frame, text="Start", state=tk.DISABLED, command=self.start_processing)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = ttk.Button(control_frame, text="Play", state=tk.DISABLED, command=self.toggle_playback)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(control_frame, text="Save", state=tk.DISABLED, command=self.save_file)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Colorizer selection
        self.colorizer_var = tk.StringVar()
        colorizer_frame = ttk.Frame(control_frame)
        colorizer_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(colorizer_frame, text="Colorizer:").pack(side=tk.LEFT)
        self.colorizer_dropdown = ttk.Combobox(
            colorizer_frame, 
            textvariable=self.colorizer_var,
            values=["ECCV16", "SIGGRAPH17"],
            state="readonly"
        )
        self.colorizer_dropdown.current(1)
        self.colorizer_dropdown.pack(side=tk.LEFT, padx=5)
        
        # GPU Checkbox
        self.use_gpu_var = tk.BooleanVar()
        self.gpu_check = ttk.Checkbutton(control_frame, text="Use GPU", variable=self.use_gpu_var)
        self.gpu_check.pack(side=tk.LEFT, padx=5)
        
        # Media displays
        self.input_label = ttk.Label(self.left_frame)
        self.input_label.pack(fill=tk.BOTH, expand=True)
        
        self.output_label = ttk.Label(self.right_frame)
        self.output_label.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')

    def toggle_playback(self):
        if self.playing:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self):
        if self.input_player and self.output_player:
            self.playing = True
            self.play_btn.config(text="Stop")
            self.input_player.start()
            self.output_player.start()

    def stop_playback(self):
        self.playing = False
        self.play_btn.config(text="Play")
        if self.input_player:
            self.input_player.stop()
        if self.output_player:
            self.output_player.stop()

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[
            ("Media Files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov *.mkv")
        ])
        
        if file_path:
            self.input_path = file_path
            self.file_type = "image" if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) else "video"
            self.show_input_media(file_path)
            self.start_btn.config(state=tk.NORMAL)
            self.play_btn.config(state=tk.NORMAL if self.file_type == "video" else tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)
            self.stop_playback()

    def show_input_media(self, file_path):
        if self.file_type == "image":
            img = Image.open(file_path)
            img.thumbnail((500, 500))
            photo = ImageTk.PhotoImage(img)
            self.input_label.config(image=photo)
            self.input_label.image = photo
        else:
            self.input_player = VideoPlayer(self.input_label, file_path)
            self.input_player.start()
            self.input_player.stop()

    def start_processing(self):
        if not self.input_path:
            return
        
        self.processing = True
        self.output_data = None
        self.start_btn.config(state=tk.DISABLED)
        self.play_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.progress.pack(side=tk.LEFT, padx=5)
        self.progress.start()
        
        threading.Thread(target=self.process_media).start()
        self.root.after(100, self.check_processing)

    def process_media(self):
        try:
            colorizer = app.colorizer_eccv16 if self.colorizer_var.get() == "ECCV16" else app.colorizer_siggraph17
            
            if self.file_type == "image":
                self.output_data = app.colorize_image(
                    self.input_path, 
                    colorizer, 
                    use_gpu=self.use_gpu_var.get()
                )
            else:
                output_path = "temp_output.mp4"
                app.colorize_video(
                    self.input_path,
                    output_path,
                    colorizer,
                    use_gpu=self.use_gpu_var.get()
                )
                self.output_data = output_path
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.processing = False

    def check_processing(self):
        if self.processing:
            self.root.after(100, self.check_processing)
        else:
            self.progress.stop()
            self.progress.pack_forget()
            self.show_output_media()
            self.save_btn.config(state=tk.NORMAL)
            self.start_btn.config(state=tk.NORMAL)
            self.play_btn.config(state=tk.NORMAL if self.file_type == "video" else tk.DISABLED)
            if self.file_type == "video":
                self.output_player = VideoPlayer(self.output_label, self.output_data)
                self.output_player.start()
                self.output_player.stop()

    def show_output_media(self):
        if self.file_type == "image":
            img_output = (self.output_data * 255).astype(np.uint8)
            img = Image.fromarray(img_output)
            img.thumbnail((500, 500))
            photo = ImageTk.PhotoImage(img)
            self.output_label.config(image=photo)
            self.output_label.image = photo

    def save_file(self):
        if self.output_data is None:
            messagebox.showerror("Error", "No output to save!")
            return
        
        file_types = [
            ("PNG", "*.png"),
            ("JPEG", "*.jpg"),
            ("MP4", "*.mp4"),
            ("All Files", "*.*")
        ] if self.file_type == "image" else [
            ("MP4", "*.mp4"),
            ("All Files", "*.*")
        ]
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png" if self.file_type == "image" else ".mp4",
            filetypes=file_types
        )
        
        if save_path:
            try:
                if self.file_type == "image":
                    img_output = (self.output_data * 255).astype(np.uint8)
                    img_output = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(save_path, img_output)
                else:
                    os.rename(self.output_data, save_path)
                messagebox.showinfo("Success", "File saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    gui = ColorizationApp(root)
    root.mainloop()