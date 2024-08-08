# G2D-Diff: A genotype-to-drug diffusion model for generation of tailored anti-cancer small molecules
Official repository of the G2D-Diff: A genotype-to-drug diffusion model for generation of tailored anti-cancer small molecules. 
Diffusion source code was adopted from the https://github.com/lucidrains/denoising-diffusion-pytorch. 

# Environment setting (Anaconda)
- Create virtual environment 
> conda create -n g2d_diff python=3.8.10

- Activate environment
> conda activate g2d_diff
 
- Install required packages
> pip install -r requirement.txt --extra-index-url https://download.pytorch.org/whl/cu113

# Generation Tutorial
- GenerationTutorial.ipynb
Generation with the trained condition encoder and diffusion model.

# Reproducing the models
Use the following jupyter notebooks after add the kernel.
> python -m ipykernel --user --name g2d_diff

## For training G2D-Diff
- Single GPU
> accelerate launch --num_processes=1 --gpu_ids=0 distributed_G2D_Diff.py

- Multiple GPUs (Check available GPU IDs)
> accelerate launch --num_processes=2 --gpu_ids=0,1 distributed_G2D_Diff.py

Check the hyperparameters in the python file (distributed_G2D_Diff.py)

## For training condition encoder
- ConitionEncoderPretraining.ipynb

## For training G2D-Pred
- G2DPredTraining.ipynb







