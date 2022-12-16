# Few-shot level updating using generative models

## Data generation

One can generate regular or fine-tuning datasets, using the flag ````-finetune```. 

```
python .\src\utils\dataset_generation.py <q_mark/cannon/coin>
```

## Training the VAE model

Train a level generator that uses a UNet VAE. The training makes use of impainting masking to reduce overfitting.
The trained model is stored in wandb. 

```
python train_VAE.py unet <q_mark/cannon/coin>
```

## Testing reconstruction of the VAE model

Test the UNet VAE generator by reconstructing masked inputs.

```
python reconstruction.py <wandb_checkpoint> <q_mark/cannon/coin>
```

## Fine-tuning the VAE model

The VAE generator can be fine-tuned by loading a previously stored models and training with a dataset that includes some examples of the hidden token.

```
python finetune_VAE.py <wandb_checkpoint> <q_mark/cannon/coin>
```

## References

1. For the plotting functions and the basic architecture of the VAE (````vanilla_vae.py```), we based our work in the repository [minimal_VAE_on_Mario](https://github.com/miguelgondu/minimal_VAE_on_Mario).
2. For the Super Mario Bros. dataset, we referred to the repository [Mario-AI-Framework](https://github.com/miguelgondu/minimal_VAE_on_Mario)https://github.com/amidos2006/Mario-AI-Framework).

## Project directory

    ├── README.md                    # overview of the project  
    ├── train_VAE.py                 # trains VAE reconstructor
    ├── reconstruction.py            # compares original levels and VAE reconstructions   
    ├── train_TNet.py                # trains TNetwork updater (VAE + 2 networks that encode update examples)
    ├── update.py                    # generates updates given a trained TNetwork (which contains a trained VAE)
    ├── finetune_VAE.py              # fine-tunes the VAE reconstructor       
    ├── finetune_TNet.py             # fine-tunes the TNet updater (ideally, receives as input the fine-tuned VAE)
    ├── create_evaluation_files.py   # converts pickled texts into separate frame files
    ├── evaluation.py                # accuracy evaluation based on generated frame files     
    ├── data/                        # data files used in the project  
    └── src/                         # contains all code in the project  
        ├── LICENSE                  # MIT license  
        ├── requirements.txt         # software requirements and dependencies  
        └── level_generators  
        │   ├── base.py  
        |   └── tnet.py
        |   └── vae.py        
        └── nets  
        │   ├── bases.py 
        |   └── blocks.py 
        |   └── tnet.py
        |   └── unet_vae.py  
        |   └── vanilla_vae.py   
        └── utils
        |   └── sprites  
        |   └── data_loading.py 
        |   └── dataset_generation.py
        |   └── levels.py  
        |   └── load_models.py 
        |   └── plotting.py   
            └── training.py   
