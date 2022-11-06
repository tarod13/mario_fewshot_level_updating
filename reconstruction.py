from src.level_generators import VAEGenerator
from src.utils.data import load_data
from src.utils.plotting import get_img_from_level


def run():
    checkpoint = "./lightning_logs/version_1/checkpoints/epoch=999-step=34000.ckpt"
    mario_val = load_data()[1]    
    mario_generator = VAEGenerator.load_from_checkpoint(checkpoint)
    mario_generator.reconstruct(mario_val)

if __name__ == "__main__":
    run()
