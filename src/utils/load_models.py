import wandb
from pathlib import Path

from src.level_generators import VAEGenerator

def freeze(module):
    for param in module.parameters():
        param.requires_grad = False
        
def copy_network(target_network, source_network):
    for target_param, source_param in zip(
        target_network.parameters(), source_network.parameters()
    ):
        target_param.data.copy_(source_param.data)

def load_VAE_model(**kwargs):
    run = wandb.init(project="mario_level_updating")

    artifact = run.use_artifact(kwargs['wandb_checkpoint'], type="model")
    artifact_dir = artifact.download()

    vae_generator = VAEGenerator.load_from_checkpoint(
        Path(artifact_dir) / "model.ckpt")
    return vae_generator

def copy_VAE_into_TNet(vae_model, tnet_generator):
    copy_network(tnet_generator.tnet.unet, vae_model)
    
