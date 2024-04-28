# CoSE: Compositional Stroke Embeddings

This repository is forked from [CoSE repository](https://github.com/eth-ait/cose) for my thesis project. 


## To-do
1. Update the code to make sure the preprocessing and undoing preprocessing work as intended.
2. Finish autoregressive_prediction function.
3. Mark some scripts useful for understanding the CoSE model architecture and data preprocessing. Create a table to add brief introduction to these scripts. Some examples include:
    - data_scripts/didi_json_to_tfrecords.py
    - data_scripts/calculate_data_statistics.py
    - smartink-js/lib/stroke_predictor.js
    - smartink-js/lib/stroke_embedding.js
    - smartink/data/stroke_dataset.py
    - smartink/models/stroke/t_emb.py
    - smartink/source/eval_engine.py 
    - smartink/util/ink.py. 
4. Think about whether the [original evaluation python script](smartink/source/eval_engine.py) can be adapted to help do both quantitative and qualitative evaluation.  


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