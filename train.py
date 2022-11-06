from pytorch_lightning import Trainer

from src.level_generators import VAEGenerator
from src.utils.data import generate_dataset


def run():
    mario_train, mario_val = generate_dataset()
    mario_generator = VAEGenerator()
    trainer = Trainer()
    trainer.fit(mario_generator, mario_train, mario_val)

if __name__ == "__main__":
    run()

