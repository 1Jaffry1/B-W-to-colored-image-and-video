import cv2
import numpy as np
import torch
import colorizers
import os
from colorizers import load_img, preprocess_img, postprocess_tens

# Initialize models
colorizer_eccv16 = colorizers.eccv16(pretrained=True).eval()
colorizer_siggraph17 = colorizers.siggraph17(pretrained=True).eval()

def colorize_image(img_path, colorizer, use_gpu=False):
    print(f"Reading image from path: {img_path}")
    img = load_img(img_path)  # Load the image
    if img is None:
        raise ValueError(f"Failed to load image from path: {img_path}")
    print(f"Image successfully loaded: {img.shape}")

    # Preprocess image
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))  # Preprocess the image
    if use_gpu:
        colorizer.cuda()
        tens_l_rs = tens_l_rs.cuda()

    # Run through the model
    with torch.no_grad():
        ab_channels = colorizer(tens_l_rs).cpu()  # Get the ab channels
        print(f"Model output shape: {ab_channels.shape}")

    # Postprocess the result
    result = postprocess_tens(tens_l_orig, ab_channels)
    print(f"Postprocessing completed. Result shape: {result.shape}")
    return result

def colorize_video(video_path, output_path, colorizer, use_gpu=False):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process each frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        if use_gpu:
            colorizer.cuda()
            tens_l_rs = tens_l_rs.cuda()
        
        with torch.no_grad():
            ab_channels = colorizer(tens_l_rs).cpu()
        
        colorized_frame = postprocess_tens(tens_l_orig, ab_channels)

        # Ensure proper scaling and conversion
        colorized_frame_bgr = np.clip(colorized_frame * 255, 0, 255).astype('uint8')
        colorized_frame_bgr = cv2.cvtColor(colorized_frame_bgr, cv2.COLOR_RGB2BGR)

        out.write(colorized_frame_bgr)
    
    cap.release()
    out.release()

# Example usage
if __name__ == "__main__":
    # Define paths
    images_path = "imgs"
    out_images = "imgs_out"
    vids_path = "input_vids"
    out_vids = "out_vids"

    # Ensure output directories exist
    for directory in [vids_path, out_vids, out_images]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Process all images in the input directory
    for im_path in os.listdir(images_path):
        full_im_path = os.path.join(images_path, im_path)
        if im_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Processing: {full_im_path}")
            try:
                img_result = colorize_image(full_im_path, colorizer_siggraph17, use_gpu=False)

                # Ensure proper scaling and conversion before saving
                img_result_bgr = np.clip(img_result * 255, 0, 255).astype('uint8')
                img_result_bgr = cv2.cvtColor(img_result_bgr, cv2.COLOR_RGB2BGR)

                # Save the image
                output_path = os.path.join(out_images, os.path.basename(im_path))
                cv2.imwrite(output_path, img_result_bgr)
                print(f"Saved: {output_path}")

                # Optional: Show the image
                cv2.imshow('Colorized Image', img_result_bgr)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            except Exception as e:
                print(f"Error processing {full_im_path}: {e}")

    # Video colorization example
    # colorize_video("input_video.mp4", "output_video.mp4", colorizer_eccv16, use_gpu=False)
