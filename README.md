# G2D-Diff: A genotype-to-drug diffusion model for generation of tailored anti-cancer small molecules
## Abstract
We present Genotype-to-Drug Diffusion (G2D-Diff), a generative artificial intelligence (AI) approach for creating small molecule-based drug structures tailored to specific cancer genotypes. G2D-Diff demonstrates exceptional performance in generating diverse, drug-like compounds that meet desired efficacy conditions for a given genotype. The model outperforms existing methods in diversity, feasibility, and condition fitness. G2D-Diff learns directly from drug response data distributions, ensuring reliable candidate generation without separate predictors. Its attention mechanism provides insights into potential cancer targets and pathways, enhancing interpretability. In triple-negative breast cancer case studies, G2D-Diff generated plausible hit-like candidates by focusing on relevant pathways. By combining realistic hit-like molecule generation with relevant pathway suggestions for specific genotypes, G2D-Diff represents a significant advance in AI-guided, personalized drug discovery. This approach has the potential to accelerate drug development for challenging cancers by streamlining hit identification, and lead optimization. 

![g2d_diff_fig](https://github.com/user-attachments/assets/8feec49b-dc43-4c9b-8e73-2f314524c893)


## Information
Official repository of the G2D-Diff: A genotype-to-drug diffusion model for generation of tailored anti-cancer small molecules.  

Diffusion source code was adapted from the https://github.com/lucidrains/denoising-diffusion-pytorch.   
All software dependencies are listed in "requirement.txt" text file.

Contact Info:   
hjnam@gist.ac.kr  
hyunhokim@gm.gist.ac.kr

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
It will take about 15 minutes for a single genotype input (ex. a cell line), but the time may vary depending on the environment.  
Check the comments in the notebook for further information about the source code.  
(ex. saving checkpoints. You may need to create the directory for saving.)

# Reproducing the models
Use the following jupyter notebooks after adding the kernel. 
> python -m ipykernel --user --name g2d_diff

## For training G2D-Diff
- Single GPU
> accelerate launch --num_processes=1 --gpu_ids=0 distributed_G2D_Diff.py

- Multiple GPUs (Check available GPU IDs)
> accelerate launch --num_processes=2 --gpu_ids=0,1 distributed_G2D_Diff.py

Check the comments in the python file for further information about the source code (distributed_G2D_Diff.py, ex. saving checkpoints).

## For training condition encoder
- ConitionEncoderPretraining.ipynb  
Check the comments in the notebook for further information about the source code (ex. saving checkpoints).
 
## For training G2D-Pred
- G2DPredTraining.ipynb  
Check the comments in the notebook for further information about the source code (ex. saving checkpoints).






