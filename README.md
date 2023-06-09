# RaRetina
Deep Learning algorithm for neurological segmentation of Optical Coherence Tomography images of human retina

![alt text](./example/prediction.png)


Running the code:

CUDA_VISIBLE_DEVICES=0 python main.py --mode train --model_name transunet



# RaRetina

This is the codebase for Intra-retinal layer segmentation of optical coherent tomography images using deep learning

# Usage

This section of the README walks through how to train and sample from a model.

## Installation

Clone this repository and navigate to it in your terminal. Then run:

```
conda env create -f environment.yml
conda activate torch-kernel
```

This should install the `torch-kernel` conda package that the scripts depend on.


## Preparing Data

The training code reads images from a directory of image files. In the config file, we have provided path to the directory of [Duke People Dataset](https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm)


## Training

To train your model, you should first decide some hyperparameters depending on the architecture that is being used - image dimensions that is being used. Cosine learning scheduler could be adjusted if one wants.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --mode train --model_name unet
```


 * **Parallel comutation:** add `CUDA_VISIBLE_DEVICES=0,1,2,...` if training across multiple units of GPU
 * **Model name:** change `unet` to `resnetunet` or `transunet` 


## Sampling

The above training script saves checkpoints to `.pth` files in the logging directory. These checkpoints will have names like `ResNext50_39_epoch.pth`

Once you have a path to your model, you can generate a large batch of samples like so:

```
python main.py --mode inference --model_name resnetunet --model_path /vol/ResNext50_39_epoch.pth --img_path /vol/102651.tif
```

## Example

Deep Learning algorithm for neurological segmentation of Optical Coherence Tomography images of human retina

![alt text](./example/prediction.png)

