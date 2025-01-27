import cv2
import numpy as np
import torch
import colorizers
import os
from colorizers import load_img, preprocess_img, postprocess_tens

# Initialize models
colorizer_eccv16 = colorizers.eccv16(pretrained=True).eval() # this model is fast, but produces less accurate and less realistic results, we use this model to convert videos.
colorizer_siggraph17 = colorizers.siggraph17(pretrained=True).eval() # this model is slower, but produces much more accurate and realistic results. We use this model to convert the images


# preproccessing function is defined in colorizers/util.py, the docs are availble there. 

def colorize_image(img_path, colorizer, use_gpu=False):
    print(f"Reading image from path: {img_path}")
    img = load_img(img_path)  # Load the image
    if img is None:
        raise ValueError(f"Failed to load image from path: {img_path}")
    print(f"Image successfully loaded: {img.shape}")

    # Preprocess image
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))  # Preprocess the image (resize, convert to LAB)
    if use_gpu:
        colorizer.cuda() # makes use of GPU acceleration if GPU is available
        tens_l_rs = tens_l_rs.cuda()

    # Run through the model
    with torch.no_grad(): # Disable gradient tracking, this reduces memory consumption for computations
        ab_channels = colorizer(tens_l_rs).cpu()  # Get the ab channels
        print(f"Model output shape: {ab_channels.shape}") # Print the shape of the output

    # Postprocess the result
    result = postprocess_tens(tens_l_orig, ab_channels) # resize back the predicted ab channels and concatenate with the original L channel
    print(f"Postprocessing completed. Result shape: {result.shape}")
    return result

def colorize_video(video_path, output_path, colorizer, use_gpu=False):
    """
    Colorizes a black and white video using a given colorizer model.
    Args:
        video_path (str): Path to the input black and white video file.
        output_path (str): Path to save the colorized output video file.
        colorizer (torch.nn.Module): Pre-trained colorizer model to use for colorizing frames.
        use_gpu (bool, optional): Flag to indicate whether to use GPU for processing. Defaults to False.
    Returns:
        None, saves the video to the output path.
    """
    cap = cv2.VideoCapture(video_path) # Initialize the video capture object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Define the codec for the output video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) # Initialize the output video writer
    
    while cap.isOpened(): # cap is the video capture object
        ret, frame = cap.read()  # Read the next frame
        if not ret: # If the frame is not read properly, break the loop (end of video)
            break

        # Process each frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256)) # Preprocess the frame
        if use_gpu: # Use GPU if available
            colorizer.cuda()
            tens_l_rs = tens_l_rs.cuda()
        
        with torch.no_grad():
            ab_channels = colorizer(tens_l_rs).cpu() # Runs the specified model (colorizer = SIGGRAPH17) on the frame
        
        colorized_frame = postprocess_tens(tens_l_orig, ab_channels) # Concatenate the original L channel with the predicted ab channels

        # Ensure proper scaling and conversion
        colorized_frame_bgr = np.clip(colorized_frame * 255, 0, 255).astype('uint8') # Clip the values to the range [0, 255] and convert to uint8
        colorized_frame_bgr = cv2.cvtColor(colorized_frame_bgr, cv2.COLOR_RGB2BGR) # Convert the color space from RGB to BGR

        out.write(colorized_frame_bgr)
    
    cap.release()
    out.release()

# Example usage
if __name__ == "__main__":
    import argparse # to get arguments from the command line ('--mode : images|videos / not required', '--use_gpu')
    import warnings 

    # Suppress warnings
    warnings.filterwarnings("ignore") 

    parser = argparse.ArgumentParser(description="Colorize images or videos using pretrained models.") # Create an argument parser, recieves args from cmd line 
    parser.add_argument("--mode", choices=["images", "videos"], required=False,
                        help="Choose to process images, videos, or both.")
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU for faster processing.")
    args = parser.parse_args() # if no mode is specified, both images and videos will be processed

    # Define paths
    images_path = "imgs"
    out_images = "imgs_out"
    vids_path = "input_vids"
    out_vids = "out_vids"

    # Ensure output directories exist
    for directory in [vids_path, out_vids, out_images]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    if args.mode in ["images", None]:
        # Process all images in the input directory
        for im_path in os.listdir(images_path): # loop through all files in the input images directory
            full_im_path = os.path.join(images_path, im_path) # get the full path of the image
            if im_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')): # check if the file is an image (just in case)
                print(f"Processing image: {full_im_path}") # print the path of the image
                try:
                    img_result = colorize_image(full_im_path, colorizer_siggraph17, use_gpu=args.use_gpu) # colorize the image using the SIGGRAPH17 model
                    output_path = os.path.join(out_images, os.path.basename(im_path)) # define the output path
                    cv2.imwrite(output_path, cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)) # save the colorized image to the output path
                    print(f"Image saved: {output_path}") # print? success!
                
                except Exception as e:
                    print(f"Error processing {full_im_path}: {e}")

    if args.mode in ["videos", None]: # if the mode is videos or not specified
        # Process all videos in the input directory
        for vid_path in os.listdir(vids_path):  # loop through all files in the input videos directory
            full_vid_path = os.path.join(vids_path, vid_path) # get the full path of the video
            if vid_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.mpeg')): # check if the file is a video (just in case)
                print(f"Processing video: {full_vid_path}") # print the path of the video
                try:
                    output_vid_path = os.path.join(out_vids, os.path.basename(vid_path)) # define the output path
                    colorize_video(full_vid_path, output_vid_path, colorizer_eccv16, use_gpu=args.use_gpu) # colorize the video using the ECCV16 model
                    print(f"Video saved: {output_vid_path}") # print? success!
                except Exception as e:
                    print(f"Error processing {full_vid_path}: {e}")



