# ReadingEmebdding - Implementation
### From Word Embedding to Reading Embedding Using Large Language Model, EEG and Eye-tracking

Accepted to [IEEE-EMBC 2024](https://embc.embs.org/2024/)

Paper : https://arxiv.org/pdf/2401.15681.pdf


![The Overall Workflow](./images/scheme.png?raw=true "Title")

Abstract: Reading comprehension, a fundamental cognitive ability essential for knowledge acquisition, is a complex skill, with a notable number of learners lacking proficiency in this domain.

This study introduces innovative tasks for Brain-Computer Interface (BCI), predicting the relevance of words or tokens read by individuals to the target inference words. We use state-of-the-art Large Language Models (LLMs) to guide a new reading embedding representation in training that integrates EEG and eye-tracking biomarkers through an attention-based encoder.

This study pioneers the integration of LLMs, EEG, and eye-tracking for predicting human reading comprehension at the word level.

## Requirements
Implemented in `Python3.10` with the following key packages:
```shell
pytorch = 2.0.1
scikit-learn = 2.1.2
numpy = 1.25.0
scipy = 1.10.1

# For plotting
matplotlib
seaborn
```
## Datasets
Pre-processed from [ZuCo 1.0](https://www.nature.com/articles/sdata2018291): [Google Drive](https://drive.google.com/drive/folders/1c8qsZtEcA5zUQOwcpqS90LBHzIBTOyns?usp=sharing) Download and keep them in `./Datasets/`

## Usage (Will be updated with easier hyper-parameter settings)
- `trainREmodel.py` to train the model on the datasets
- `CV_REmodel.py` to perform K-fold cross validation on the datasets 

*Sample `SLURM` script provided in `script_train.sh` if needed to run on a cluster*

- `TransformerClassifier_REmbedding.ipynb` for an overview of code in an all-in-one style

## Citation
Cite using the Bibtex citation below
```LaTeX
@article{zhang2024word,
  title={From Word Embedding to Reading Embedding Using Large Language Model, EEG and Eye-tracking},
  author={Zhang, Yuhong and Yang, Shilai and Cauwenberghs, Gert and Jung, Tzyy-Ping},
  journal={arXiv preprint arXiv:2401.15681},
  year={2024}
}
```
