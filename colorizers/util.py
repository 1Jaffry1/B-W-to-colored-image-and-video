
from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed

def load_img(img_path): # load an image from a given path, and return it as a numpy array
	out_np = np.asarray(Image.open(img_path)) # read the image from the path
	if(out_np.ndim==2): # if the image is grayscale, convert it to RGB
		out_np = np.tile(out_np[:,:,None],3) # add singleton dimension to the last axis, and tile it 3 times, what is singleton? It is a dimension with size 1
	return out_np # return the image as a numpy array

def resize_img(img, HW=(256,256), resample=3):
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample)) # resize the image to the given dimensions (default: 256x256)

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3): # img_rgb_orig is the original image in RGB format 
	# return original size L and resized L as torch Tensors
	img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample) # resize the image to the given dimensions, resample=3 means using the bicubic interpolation, upscaling/downscaling the image based on the pixels around it
	
	img_lab_orig = color.rgb2lab(img_rgb_orig) # convert the image to LAB color space
	img_lab_rs = color.rgb2lab(img_rgb_rs) # convert the resized image to LAB color space

	img_l_orig = img_lab_orig[:,:,0] # extract the L channel from the original image
	img_l_rs = img_lab_rs[:,:,0] # extract the L channel from the resized image
	
	# add singleton dimensions to the tensors
	tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:] 
	tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:] 

	return (tens_orig_l, tens_rs_l) # return the original size L and resized L as torch Tensors, we need the original size L for postprocessing

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'): # resize back the predicted ab channels and concatenate with the original L channel
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

	HW_orig = tens_orig_l.shape[2:] # get the dimensions of the original L channel
	HW = out_ab.shape[2:] # get the dimensions of the predicted ab channels

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]): # if the dimensions of the original L channel and the predicted ab channels are different
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear') # resize the predicted ab channels to the dimensions of the original L channel
	else:
		out_ab_orig = out_ab # if the dimensions are the same, no need to resize

	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1) # concatenate the original L channel with the resized predicted ab channels
	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0))) # convert the concatenated image to RGB and return it
