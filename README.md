# PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models

Code accompanying CVPR'20 paper of the same title. Paper link: https://arxiv.org/abs/2003.03808

![Transformation Preview](./readme_resources/014.jpeg)
![Transformation Preview](./readme_resources/034.jpeg)
![Transformation Preview](./readme_resources/094.jpeg)

Table of Contents
=================
- [PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models](#pulse-self-supervised-photo-upsampling-via-latent-space-exploration-of-generative-models)
- [Table of Contents](#table-of-contents)
  - [What does it do?](#what-does-it-do)
  - [How do I use it?](#how-do-i-use-it)
  - [Usage](#usage)
    - [Prereqs](#prereqs)
    - [Data](#data)
    - [Applying PULSE](#applying-pulse)
## What does it do? 
Given a low-resolution input image, PULSE searches the outputs of a generative model (here, [StyleGAN](https://github.com/NVlabs/stylegan)) for high-resolution images that are perceptually realistic and downscale correctly.

![Transformation Preview](./readme_resources/transformation.gif)

## How do I use it?

The easiest way to apply PULSE to your own images is with our interactive demo, found at https://colab.research.google.com/drive/1-cyGV0FoSrHcQSVq3gKOymGTMt0g63Xc?usp=sharing#sandboxMode=true. 

If you want to try using this codebase, continue on.


## Usage

The main file of interest for applying PULSE is `run.py`. A full list of arguments with descriptions can be found in that file; here we describe those relevant to getting started.

### Prereqs

You will need to install cmake first (required for dlib, which is used for face alignment). Currently the code only works with CUDA installed (and therefore requires an appropriate GPU) and has been tested on Linux. For the full set of required Python packages, create a Conda environment from the provided YAML, e.g.
```
conda create -f pulse.yml
```
Finally, you will need an internet connection the first time you run the code as it will automatically download the relevant pretrained model from Google Drive (if it has already been downloaded, it will use the local copy).

### Data

By default, input data for `run.py` should be placed in `./input/` (though this can be modified). However, this assumes faces have already been aligned and downscaled. If you have data that is not already in this form, place it in `realpics` and run `align_face.py` which will automatically do this for you. (Again, all directories can be changed by command line arguments if more convenient.) You will at this stage pic a downscaling factor. 

Note that if your data begins at a low resolution already, downscaling it further will retain very little information. In this case, you may wish to bicubically upsample (usually, to 1024x1024) and allow `align_face.py` to downscale for you.  

The dataset we evaluated on was [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans), but in our experience PULSE works with any picture of a realistic face.

### Applying PULSE
Once your data is appropriately formatted, all you need to do is
```
python run.py
```
Enjoy!

*Contact both Sachit Menon and Alex Damian (sachit.menon@duke.edu and alexandru.damian@duke.edu) for questions regarding this work.*