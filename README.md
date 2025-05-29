# G2D-Diff: A genotype-to-drug diffusion model for generation of tailored anti-cancer small molecules
## Information
[![DOI](https://zenodo.org/badge/832038278.svg)](https://doi.org/10.5281/zenodo.15265966)

Official repository of the G2D-Diff: A genotype-to-drug diffusion model for generation of tailored anti-cancer small molecules.  

All software dependencies are listed in "requirement.txt" text file.

Contact Info:   
hjnam@gist.ac.kr  
hyunhokim@gm.gist.ac.kr

### Abstract
Despite advances in precision oncology, developing effective cancer therapeutics remains a significant challenge due to tumor heterogeneity and the limited availability of well-defined drug targets. Recent progress in generative artificial intelligence (AI) offers a promising opportunity to address this challenge by enabling the design of hit-like anti-cancer molecules conditioned on complex genomic features. We present Genotype-to-Drug Diffusion (G2D-Diff), a generative AI approach for creating small molecule-based drug structures tailored to specific cancer genotypes. G2D-Diff demonstrates exceptional performance in generating diverse, drug-like compounds that meet desired efficacy conditions for a given genotype. The model outperforms existing methods in diversity, feasibility, and condition fitness. G2D-Diff learns directly from drug response data distributions, ensuring reliable candidate generation without separate predictors. Its attention mechanism provides insights into potential cancer targets and pathways, enhancing interpretability. In triple-negative breast cancer case studies, G2D-Diff generated plausible hit-like candidates by focusing on relevant pathways. By combining realistic hit-like molecule generation with relevant pathway suggestions for specific genotypes, G2D-Diff represents a significant advance in AI-guided, personalized drug discovery. This approach has the potential to accelerate drug development for challenging cancers by streamlining hit identification.

![g2d_diff_fig](https://github.com/GIST-CSBL/G2D-Diff/blob/main/Fig1_LastRev_2.png)


# Environment setting (Linux (Ubuntu) and Anaconda)
- Create virtual environment 
> conda create -n g2d_diff python=3.8.10

- Activate environment
> conda activate g2d_diff
 
- Install required packages
> pip install -r requirement.txt --extra-index-url https://download.pytorch.org/whl/cu113

The installation typically takes around 10 minutes, but the time may vary depending on the environment.

# IMPORTANT!
You can download all processed datasets, model checkpoints in this google drive link.  
https://drive.google.com/file/d/1qk4Wwkqvwas7kpjcuFKbSCT8aPaP8RKI/view?usp=drive_link  
You must unpack this zip file in the repository folder.  
In G2D-Diff directory...    
  -- data <- Need to download from the link above.  
  -- src  
  -- other files...    
If you have any problem in downloading the data and model checkpoints, feel free to ask me by email (hyunho.kim@kitox.re.kr).  

# Generation tutorial
- GenerationTutorial.ipynb
 
Generation with the trained condition encoder and diffusion model.  
It will take about 15 minutes for a single genotype input (ex. a cell line), but the time may vary depending on the environment.  

Check the comments in the notebook for further information about the source code.  
(ex. saving checkpoints. You may need to create a directory for saving.)

# Reproducing the models
Use the following jupyter notebooks after adding the kernel. 
> python -m ipykernel --user --name g2d_diff


For all .py file and jupyter notebooks for reproducing the models, check the comments for further information.  
(ex. saving checkpoints. You may need to create a directory for saving.)

### For training G2D-Diff
- Single GPU
> accelerate launch --num_processes=1 --gpu_ids=0 distributed_G2D_Diff.py

- Multiple GPUs (Check available GPU IDs)
> accelerate launch --num_processes=2 --gpu_ids=0,1 distributed_G2D_Diff.py

### For training condition encoder
- ConitionEncoderPretraining.ipynb

 
### For training G2D-Pred
- G2DPredTraining.ipynb
- G2DPredTrainingFromScratch.ipynb




## License

This repository contains materials under two different licenses:

### Code License (PolyForm Noncommercial License 1.0.0)
All source code in this repository is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/), which permits use for non-commercial purposes only.  
See the [LICENSE](LICENSE) file for full terms.

### Data & Model License (CC BY-NC-SA 4.0)
The trained model weights and any generated data are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).  
See the [DATA_LICENSE](DATA_LICENSE) for details.

---

### Third-party Notice (MIT License)
Parts of this codebase are adapted from [Phil Wang's denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch), which is licensed under the MIT License. The relevant files retain proper attribution and include the original license text as required.


Last modified: 2025-05-29
