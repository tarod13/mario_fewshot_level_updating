from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer

from level_generators import VAEGenerator
from vae_base import load_data


def generate_dataset(batch_size: int = 64):
    # Load data
    train_tensors, val_tensors = load_data()

    # Create dataloaders
    train_dataset = TensorDataset(train_tensors)
    mario_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_tensors)
    mario_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return mario_train, mario_val

def run():
    mario_train, mario_val = generate_dataset()
    mario_generator = VAEGenerator()
    trainer = Trainer()
    trainer.fit(mario_generator, mario_train, mario_val)

if __name__ == "__main__":
    run()

