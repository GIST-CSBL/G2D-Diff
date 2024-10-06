# G2D-Diff: A genotype-to-drug diffusion model for generation of tailored anti-cancer small molecules
Official repository of the G2D-Diff: A genotype-to-drug diffusion model for generation of tailored anti-cancer small molecules.
 
Diffusion source code was adapted from the https://github.com/lucidrains/denoising-diffusion-pytorch. 

All software dependencies are listed in "requirement.txt" text file.

# Environment setting (Anaconda)
- Create virtual environment 
> conda create -n g2d_diff python=3.8.10

- Activate environment
> conda activate g2d_diff
 
- Install required packages
> pip install -r requirement.txt --extra-index-url https://download.pytorch.org/whl/cu113

The installation typically takes around 10 minutes, but the time may vary depending on the environment.

# Generation Tutorial
- GenerationTutorial.ipynb
 
Generation with the trained condition encoder and diffusion model.

It will take about 15 minutes, but the time may vary depending on the environment.

Check the comments in the notebook for further information about the source code.

# Reproducing the models
Use the following jupyter notebooks after adding the kernel. 
> python -m ipykernel --user --name g2d_diff

## For training G2D-Diff
- Single GPU
> accelerate launch --num_processes=1 --gpu_ids=0 distributed_G2D_Diff.py

- Multiple GPUs (Check available GPU IDs)
> accelerate launch --num_processes=2 --gpu_ids=0,1 distributed_G2D_Diff.py

It will take about a week with four gpus, but the time may vary depending on the environment.

Check the comments in the python file for further information about the source code (distributed_G2D_Diff.py)

## For training condition encoder
- ConitionEncoderPretraining.ipynb

It will take a few hours, but the time may vary depending on the environment.

Check the comments in the notebook for further information about the source code.
 
## For training G2D-Pred
- G2DPredTraining.ipynb
  
It will take about one or two days, but the time may vary depending on the environment.

Check the comments in the notebook for further information about the source code.






