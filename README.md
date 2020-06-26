# PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models
Code accompanying CVPR'20 paper of the same title. Paper link: https://drive.google.com/file/d/1fV7FsmunjDuRrsn4KYf2Efwp0FNBtcR4/view

## NOTE

We have noticed a lot of concern that PULSE will be used to identify individuals whose faces have been blurred out. We want to emphasize that this is impossible - **PULSE makes imaginary faces of people who do not exist, which should not be confused for real people.** It will **not** help identify or reconstruct the original image.

We also want to address concerns of bias in PULSE. **We have now included a new section in the [paper](https://drive.google.com/file/d/1fV7FsmunjDuRrsn4KYf2Efwp0FNBtcR4/view) and an accompanying model card directly addressing this bias.**

---

![Transformation Preview](./readme_resources/014.jpeg)
![Transformation Preview](./readme_resources/034.jpeg)
![Transformation Preview](./readme_resources/094.jpeg)

Table of Contents
=================
- [PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models](#pulse-self-supervised-photo-upsampling-via-latent-space-exploration-of-generative-models)
- [Table of Contents](#table-of-contents)
  - [What does it do?](#what-does-it-do)
  - [Usage](#usage)
    - [Prereqs](#prereqs)
    - [Data](#data)
    - [Applying PULSE](#applying-pulse)
## What does it do? 
Given a low-resolution input image, PULSE searches the outputs of a generative model (here, [StyleGAN](https://github.com/NVlabs/stylegan)) for high-resolution images that are perceptually realistic and downscale correctly.

![Transformation Preview](./readme_resources/transformation.gif)

## Usage

The main file of interest for applying PULSE is `run.py`. A full list of arguments with descriptions can be found in that file; here we describe those relevant to getting started.

### Installation

#### Manually

You will need to install cmake first (required for dlib, which is used for face alignment). Currently the code only works with CUDA installed (and therefore requires an appropriate GPU) and has been tested on Linux and Windows. For the full set of required Python packages, create a Conda environment from the provided YAML, e.g.

```
conda create -f pulse.yml 
```
or (Anaconda on Windows):
```
conda env create -n pulse -f pulse.yml
conda activate pulse
```

In some environments (e.g. on Windows), you may have to edit the pulse.yml to remove the version specific hash on each dependency and remove any dependency that still throws an error after running ```conda env create...``` (such as readline)
```
dependencies
  - blas=1.0=mkl
  ...
```
to
```
dependencies
  - blas=1.0
 ...
```

Finally, you will need an internet connection the first time you run the code as it will automatically download the relevant pretrained model from Google Drive (if it has already been downloaded, it will use the local copy). In the event that the public Google Drive is out of capacity, add the files to your own Google Drive instead; get the share URL and replace the ID in the https://drive.google.com/uc?=ID links in ```align_face.py``` and ```PULSE.py``` with the new file ids from the share URL given by your own Drive file.

#### Using Docker
 1. [Install Docker](https://docs.docker.com/get-docker/)
 2. Verify the content of `Dockerfile` in the repo, update the first line `FROM nvidia/cuda:10.2-runtime` according to the Cuda version on your computer.
 3. From the repo folder, execute `docker build -t pulse:latest .`
 4. All the times you need to run the pulse code:
 * From the repo folder, execute `docker run -it --rm --gpus all --name pulse --entrypoint bash -v "$PWD:/home/pulse" pulse:latest`. The "-v" synchronizes your local folder "pulse" to "/home/pulse" in the docker so that you can easily have your tests pictures ("inputs" or "realpics") inside the docker. You might need to change `--gpus all` by `--runtime=nvidia` if your Docker version is older.
 * Once you are in the docker container, activate the python env `conda activate pulse`
 * Run pulse as usual (see following sections)

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
