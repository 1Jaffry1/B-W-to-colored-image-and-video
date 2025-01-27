
# Image and Video Colorization using Deep Learning

This project provides a Python-based pipeline for colorizing black-and-white images and videos using pre-trained deep learning models. The models used are `eccv16` (faster but less accurate) and `siggraph17` (slower but more accurate). The project is built using PyTorch, OpenCV, and other standard libraries.

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Colorizing Images](#colorizing-images)
   - [Colorizing Videos](#colorizing-videos)
4. [Models](#models)
5. [Directory Structure](#directory-structure)
6. [License](#license)

## Features
- Colorize black-and-white images with high accuracy using the `siggraph17` model.
- Colorize videos efficiently using the `eccv16` model.
- GPU support for faster processing.
- Easy-to-use command-line interface.

## Installation

### Prerequisites
- Python 3.8 or higher (We used 3.11 / 3.13 will not word as PyTorch does not support it)
- PyTorch
- OpenCV
- NumPy
- PIL (Pillow)
- scikit-image

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/1Jaffry1/B-W-to-colored-image-and-video.git
   cd B-W-to-colored-image-and-video
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create the `input_vids` directory for storing black-and-white videos:
   ```bash
   mkdir input_vids
   ```
Models are included in the `models` directory.

## Usage

### Colorizing Images
To colorize images **only**, place your black-and-white images in the `imgs/` directory and run the following command:
```bash
python main.py --mode images 
```
- The colorized images will be saved in the `imgs_out/` directory.
- Add the `--use_gpu` flag to enable GPU acceleration.

### Colorizing Videos
To colorize videos **only**, place your black-and-white videos in the `input_vids/` directory and run the following command:
```bash
python main.py --mode videos
```
- The colorized videos will be saved in the `out_vids/` directory.
- Add the `--use_gpu` flag to enable GPU acceleration.

## Models
- **ECCV16**: A faster model optimized for video colorization. It produces decent results but may lack fine details.
- **SIGGRAPH17**: A slower model optimized for image colorization. It produces highly accurate and realistic results.

## Directory Structure
```
project-root/
├── imgs/                  # Input black-and-white images
├── imgs_out/              # Output colorized images
├── input_vids/            # Input black-and-white videos. You need to create it.
├── out_vids/              # Output colorized videos, will be created at runtime
├── app.py                # Main script for colorization
├── colorizers/            # Custom module for model loading and preprocessing
│   ├── __init__.py        # Just a bunch of imports
│   ├── base_color.py      # Normalization and tensor conversion functions
│   ├── eccv16.py          # ECCV16 model
│   ├── siggraph17.py      # SIGGRAPH17 model
│   ├── util.py            # Pre & Post processing functions
├── requirements.txt       # List of dependencies
└── README.md              # This file
``` 
---

You can copy and paste this into a `README.md` file in your project directory. Let me know if you need further customization!

## Image flow:
1. We first convert the image to grayscale.
2. We then resize the image to 256x256 pixels.
3. We then normalize the image.
4. We then convert the image to a tensor. (To be used by the `ECCV16`  model)
5. We then pass the image through the model, which returns the colorized image.
6. We then convert the colorized image to a numpy array.
7. We then convert the numpy array to an image.
8. We then save the colorized image.

## Video flow:
1. The video is read frame by frame.
2. Each frame is colorized using the image flow. (Using the `SIGGRAPH17` model)
3. The colorized frame is saved to the output video.



### Credits:

 [[Project Page]](http://richzhang.github.io/colorization/) <br>
[Richard Zhang](https://richzhang.github.io/), [Phillip Isola](http://web.mit.edu/phillipi/), [Alexei A. Efros](http://www.eecs.berkeley.edu/~efros/). In [ECCV, 2016](http://arxiv.org/pdf/1603.08511.pdf).

**+ automatic colorization functionality for Real-Time User-Guided Image Colorization with Learned Deep Priors, SIGGRAPH 2017!**

### Citation ###

If you find these models useful for your resesarch, please cite with these bibtexs.

```
@inproceedings{zhang2016colorful,
  title={Colorful Image Colorization},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A},
  booktitle={ECCV},
  year={2016}
}

@article{zhang2017real,
  title={Real-Time User-Guided Image Colorization with Learned Deep Priors},
  author={Zhang, Richard and Zhu, Jun-Yan and Isola, Phillip and Geng, Xinyang and Lin, Angela S and Yu, Tianhe and Efros, Alexei A},
  journal={ACM Transactions on Graphics (TOG)},
  volume={9},
  number={4},
  year={2017},
  publisher={ACM}
}
```

### Misc ###
Contact Richard Zhang at rich.zhang at eecs.berkeley.edu for any questions or comments.
