import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from tkvideo import tkvideo

import app


class ColorizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image/Video Colorization")
        self.root.geometry("1200x800")
        
        # Variables
        self.input_path = None
        self.output_data = None
        self.file_type = None
        self.processing = False
        self.colorizers = {
            "ECCV16": app.colorizer_eccv16,
            "SIGGRAPH17": app.colorizer_siggraph17
        }
        
        # Create GUI elements
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
            values=list(self.colorizers.keys()),
            state="readonly"
        )
        self.colorizer_dropdown.current(1)  # Default to SIGGRAPH17
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
        
    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[
            ("Media Files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov *.mkv")
        ])
        
        if file_path:
            self.input_path = file_path
            self.file_type = "image" if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) else "video"
            self.show_input_media(file_path)
            self.start_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.DISABLED)

    def show_input_media(self, file_path):
        if self.file_type == "image":
            img = Image.open(file_path)
            img.thumbnail((500, 500))
            photo = ImageTk.PhotoImage(img)
            self.input_label.config(image=photo)
            self.input_label.image = photo
        else:
            self.input_label.config(text="Playing input video...")
            player = tkvideo(file_path, self.input_label, loop=0, size=(500,500))
            player.play()

    def start_processing(self):
        if not self.input_path:
            return
        
        self.processing = True
        self.start_btn.config(state=tk.DISABLED)
        self.progress.pack(side=tk.LEFT, padx=5)
        self.progress.start()
        
        # Run processing in separate thread
        threading.Thread(target=self.process_media).start()
        self.root.after(100, self.check_processing)

    def process_media(self):
        try:
            colorizer = self.colorizers.get(self.colorizer_var.get(), app.colorizer_siggraph17)
            
            if self.file_type == "image":
                self.output_data = app.colorize_image(
                    self.input_path, 
                    colorizer, 
                    use_gpu=self.use_gpu_var.get()
                )
            else:
                output_path = os.path.join("temp_output.mp4")
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

    def show_output_media(self):
        if self.file_type == "image":
            # Convert float32 array to uint8 and scale to 0-255 range
            img_output = (self.output_data * 255).astype(np.uint8)
            
            # Convert to PIL Image
            img = Image.fromarray(img_output)
            
            # Resize and display
            img.thumbnail((500, 500))
            photo = ImageTk.PhotoImage(img)
            self.output_label.config(image=photo)
            self.output_label.image = photo
        else:
            self.output_label.config(text="Playing output video...")
            player = tkvideo(self.output_data, self.output_label, loop=0, size=(500,500))
            player.play()

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
                    # Convert to proper format before saving
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
