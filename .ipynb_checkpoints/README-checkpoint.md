# G2D-Diff
Official repository of the G2D-Diff: A genotype-to-drug diffusion model for generation of tailored anti-cancer small molecules. 
Diffusion source code was adopted from the https://github.com/lucidrains/denoising-diffusion-pytorch. 

# Environment setting (Anaconda)
- Create virtual environment 
> conda create -n g2d_diff python=3.8.10

- Activate environment
> conda activate g2d_diff
 
- Install required packages
> pip install -r requirement.txt --extra-index-url https://download.pytorch.org/whl/cu113

# For training G2D-Diff
- Single GPU
> accelerate launch --num_processes=1 --gpu_ids=0 distributed_G2D_Diff.py

- Multi GPU
> accelerate launch --num_processes=2 --gpu_ids=0,1 distributed_G2D_Diff.py

Need to change batch size in the python file

# For pretraining condition encoder
- ConitionEncoderPretraining.ipynb

# For training G2D-Pred
- G2DPredTraining.ipynb

# Generation Tutorial
- GenerationTutorial.ipynb
