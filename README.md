# CoSE: Compositional Stroke Embeddings

This repository is forked from [CoSE repository](https://github.com/eth-ait/cose) for my thesis project.   

## Set up the workflow

### Environmenmt
Configure the virtual environment by running the following bash command targetting at [environment configuration bash script](environment_configuration.sh):
```bash
chmod +x environment_configuration.sh
source environment_configuration.sh 
conda deactivate
```

### Pre-trained Models
The pre-trained model can be downloaded from authors' [shared Google Drive](https://drive.google.com/drive/folders/1C6m7dbXaL4wn5Z4-K7ZniqoZaNTiQBdP?usp=sharing). 


### Trained Dataset
One of the trained dataset of CoSE model is [QuickDraw dataset](https://github.com/googlecreativelab/quickdraw-dataset). Note that CoSE model requires raw files.  

## [thesis](thesis) directory


## Citation
```
@article{aksan2020cose,
  title={CoSE: Compositional Stroke Embeddings},
  author={Aksan, Emre and Deselaers, Thomas and Tagliasacchi, Andrea and Hilliges, Otmar},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}

@article{gervais2020didi,
  title={The DIDI dataset: Digital Ink Diagram data},
  author={Gervais, Philippe and Deselaers, Thomas and Aksan, Emre and Hilliges, Otmar},
  journal={arXiv preprint arXiv:2002.09303},
  year={2020}
}
```   