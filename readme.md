# Few-shot level updating using generative models

## Training the model

Train a vanilla VAE

```
python train.py
```

## Project directory

├── README.md              # overview of the project
├── train                 
├── data/                  # data files used in the project
└── src/                   # contains all code in the project
    ├── LICENSE            # MIT license
    ├── requirements.txt   # software requirements and dependencies
    └── level_generators
    |   ├── base.py
    |   └── vae.py
    └── nets
    |   └── vaes
    |       ├── base.py
    |       └── vanilla.py
    └── utils
