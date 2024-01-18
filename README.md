
# Learning Disentangled Audio Representations through Controlled Synthesis

This is the reference implementation for the paper "Learning Disentangled Audio Representations through Controlled Synthesis".

## Abstract
This paper tackles the scarcity of benchmarking data in disentangled auditory representation learning. We introduce *SynTone*, a synthetic dataset with explicit ground truth explanatory factors for evaluating disentanglement techniques. Benchmarking state-of-the-art methods on SynTone highlights its utility for method evaluation. Our results underscore strengths and limitations in audio disentanglement, motivating future research.

## Table of Contents
- [Installation](#installation)
- [Dataset Creation](#dataset-creation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)
- [Citations](#citations)
- [Contact](#contact)

## Installation

Detail the steps required to install and run your code. Include information on dependencies and environment setup.

```bash
# Example installation steps
conda env create -f environment.yml
```

## Dataset Creation

**Script:** `create.py`

This script generates the *SynTone* dataset. Describe the dataset structure and how to execute the script.

```bash
# Example usage
python create.py --sample_rate 16000 --freq_start 400 --freq_end 8000 --num_freqs 100 --amp_start 0.1 --amp_end 1.0 --num_amps 10 --file_path "./Dataset/my_dataset.npz"
```

## Model Training

**Script:** `trainvae.py`

Instructions on how to train models. Include details about hyperparameters, training duration, and hardware requirements.

```bash
# Example usage
python trainvae.py --dataset /path/to/dataset --epochs 100
```

## Evaluation

**Script:** `eval.py`

Guide on how to evaluate the trained models using Supervised Disentanglement Metrics. Include any necessary flags or parameters.

```bash
# Example usage
python eval.py --model /path/to/trained_model
```

## Results

## Sample reconstruction 

![VAE Based reconstruction](Figures/Original_vs_Reconstructed.gif)

## Sampling from the learnt latent space

![VAE-Based Sampling and Generation ](Figures/Generated_Samples.gif)

## VAE Encoding and Reconstruction
![VAE Encoding and Reconstruction](Figures/Original_vs_Reconstructed_VAE.png)

## VAE Sampling and decoding
![VAE Sampling](Figures/Generated_sample_VAE.png)

## VAE Latent Space Transversal
![VAE Encoding and Reconstruction](Figures/VAE_Latent_Space_Interpolation.png)

## BetaVAE Encoding and Reconstruction
![BetaVAE Encoding and Reconstruction](Figures/Original_vs_Reconstructed_BetaVAE.png)

## BetaVAE Sampling and decoding
![VAE Sampling](Figures/Generated_sample_BetaVAE.png)

## BetaVAE Latent Space Transversal
![BetaVAE Encoding and Reconstruction](Figures/BetaVAE_Latent_Space_Interpolation.png)

## FactorVAE Encoding and Reconstruction
![FactorVAE Encoding and Reconstruction](Figures/Original_vs_Reconstructed_FactorVAE.png)

## FactorVAE Sampling and decoding
![FactorVAE Sampling](Figures/Generated_sample_FactorVAE.png)

## FactorVAE Latent Space Transversal
![FactorVAE Encoding and Reconstruction](Figures/FactorVAE_Latent_Space_Interpolation.png)

## BTCVAE Encoding and Reconstruction
![BTCVAE Encoding and Reconstruction](Figures/Original_vs_Reconstructed_BTCVAE.png)

## BTCVAE Sampling and decoding
![BTCVAE Sampling](Figures/Generated_sample_BTCVAE.png)

## BTCVAE Latent Space Transversal
![BTCVAE Encoding and Reconstruction](Figures/BTCVAE_Latent_Space_Interpolation.png)


## License

Specify the license under which your project is released.



## Citation

If you use this code in your research, please cite the paper:

```
@inproceedings{mypaper,
  title={Learning Disentangled Audio Representations through Controlled Synthesis},
  author={Authors},
  booktitle={ICLR},
  year={2024}
}
```

## Contact

