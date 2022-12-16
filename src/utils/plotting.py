"""
This script defines some functions that
plot levels.
"""
import os
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL

from src.utils.levels import onehot_to_levels

filepath = Path(__file__).parent.resolve()
Tensor = torch.Tensor


def absolute(path_str):
    return str(Path(path_str).absolute())


encoding = {
    "X": 0,
    "S": 1,
    "-": 2,
    "?": 3,
    "Q": 4,
    "E": 5,
    "<": 6,
    ">": 7,
    "[": 8,
    "]": 9,
    "o": 10,
    "b": 11,
    "B": 12,
    "m": 13,
}

sprites = {
    encoding["X"]: absolute(f"{filepath}/sprites/stone.png"),
    encoding["S"]: absolute(f"{filepath}/sprites/breakable_stone.png"),
    encoding["?"]: absolute(f"{filepath}/sprites/question.png"),
    encoding["Q"]: absolute(f"{filepath}/sprites/depleted_question.png"),
    encoding["E"]: absolute(f"{filepath}/sprites/goomba.png"),
    encoding["<"]: absolute(f"{filepath}/sprites/left_pipe_head.png"),
    encoding[">"]: absolute(f"{filepath}/sprites/right_pipe_head.png"),
    encoding["["]: absolute(f"{filepath}/sprites/left_pipe.png"),
    encoding["]"]: absolute(f"{filepath}/sprites/right_pipe.png"),
    encoding["o"]: absolute(f"{filepath}/sprites/coin.png"),
    encoding["b"]: absolute(f"{filepath}/sprites/cannon_base.png"),
    encoding["B"]: absolute(f"{filepath}/sprites/cannon.png"),
    encoding["m"]: absolute(f"{filepath}/sprites/mask.png"),
}

inverse_encoding = {
    encoding["X"]: 'X',
    encoding["S"]: 'S',
    encoding["?"]: '?',
    encoding["Q"]: 'Q',
    encoding["E"]: 'E',
    encoding["<"]: '<',
    encoding[">"]: '>',
    encoding["["]: '[',
    encoding["]"]: ']',
    encoding["o"]: 'o',
    encoding["b"]: 'b',
    encoding["B"]: 'B',
    encoding["-"]: '-',
    encoding["m"]: 'm',
}


def save_level_from_array(path, level, title=None, dpi=150):
    # Assuming that the level is a bunch of classes.
    image = get_img_from_level(level)
    plt.imshow(255 * np.ones_like(image))  # White background
    plt.imshow(image)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close()


def plot_level_from_array(ax, level, title=None):
    # Assuming that the level is a bunch of classes.
    image = get_img_from_level(level)
    ax.imshow(255 * np.ones_like(image))  # White background
    ax.imshow(image)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)


def get_img_from_level(level: np.ndarray) -> np.ndarray:
    image = []
    for row in level:
        image_row = []
        for c in row:
            if c == encoding["-"]:  # There must be a smarter way than hardcoding this.
                # white background
                tile = (255 * np.ones((16, 16, 3))).astype(int)
            elif c == -1:
                # masked
                tile = (128 * np.ones((16, 16, 3))).astype(int)
            else:
                tile = np.asarray(PIL.Image.open(sprites[c]).convert("RGB")).astype(int)
            image_row.append(tile)
        image.append(image_row)

    image = [np.hstack([tile for tile in row]) for row in image]
    image = np.vstack([np.asarray(row) for row in image])

    return image

def get_str_from_level(level: np.ndarray) -> np.ndarray:
    str_ = ''
    for row in level:
        str_row = ''
        for c in row:
            token_char = inverse_encoding[c]
            str_row += token_char
        str_ += str_row + '\n'

    return str_


def plot_level_from_decoded_tensor(dec: Tensor, ax):
    """
    Plots decoded tensor as level in ax.
    Expects {dec} to have a batch component.
    """
    level = onehot_to_levels(dec.detach().numpy())[0]
    plot_level_from_array(ax, level)
