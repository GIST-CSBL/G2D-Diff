# G2D-Diff: A genotype-to-drug diffusion model for generation of tailored anti-cancer small molecules
## Abstract
We present Genotype-to-Drug Diffusion (G2D-Diff), a generative artificial intelligence (AI) approach for creating small molecule-based drug structures tailored to specific cancer genotypes. G2D-Diff demonstrates exceptional performance in generating diverse, drug-like compounds that meet desired efficacy conditions for a given genotype. The model outperforms existing methods in diversity, feasibility, and condition fitness. G2D-Diff learns
29 directly from drug response data distributions, ensuring reliable candidate generation without separate predictors. Its attention mechanism provides insights into potential cancer targets and pathways, enhancing interpretability. In triple-negative breast cancer case studies, G2D32 Diff generated plausible hit-like candidates by focusing on relevant pathways. By combining realistic hit-like molecule generation with relevant pathway suggestions for specific genotypes, G2D-Diff represents a significant advance in AI-guided, personalized drug discovery. This approach has the potential to accelerate drug development for challenging cancers by streamlining hit identification, and lead optimization. 

## Information
Official repository of the G2D-Diff: A genotype-to-drug diffusion model for generation of tailored anti-cancer small molecules.
 
Diffusion source code was adapted from the https://github.com/lucidrains/denoising-diffusion-pytorch. 

All software dependencies are listed in "requirement.txt" text file.

Contact: 
hjnam@gist.ac.kr
hyunhokim@gm.gist.ac.kr
hyunho.kim@kitox.re.kr

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






