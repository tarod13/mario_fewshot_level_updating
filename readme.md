# Few-shot level updating using generative models

## Training the model

Train a level generator that uses vanilla VAE.

```
python train.py
```

## Testing the model

Test the vanilla VAE generator by generating reconstructions of a given input.

```
python reconstruction.py
```

## Project directory

    ├── README.md              # overview of the project  
    ├── train.py               # trains level updater
    ├── reconstruction.py      # compares original levels and reconstructions                 
    ├── data/                  # data files used in the project  
    └── src/                   # contains all code in the project  
        ├── LICENSE            # MIT license  
        ├── requirements.txt   # software requirements and dependencies  
        └── level_generators  
        │   ├── base.py  
        |   └── vae.py  
        └── nets  
        |   └── vaes  
        |       ├── base.py  
        |       └── vanilla.py  
        └── utils  
