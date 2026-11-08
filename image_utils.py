"""image_utils.py -- Lab 03: turn an image file into a fixed-length feature vector."""
import numpy as np
from PIL import Image


def image_to_features(path, size=(64, 64)):
    """Load, resize, grayscale and flatten an image into a 1-D feature vector.

    Parameters
    ----------
    path : str
        Path to an image file.
    size : tuple[int, int]
        Target (width, height). Every image is forced to this size so that
        every returned vector has the SAME length, which is what a model
        with a fixed number of input features requires.

    Returns
    -------
    numpy.ndarray of shape (size[0] * size[1],), dtype uint8
    """
    img = Image.open(path).convert("RGB")   # 1. same number of channels for every file
    img = img.resize(size)                  # 2. same width and height for every file
    img = img.convert("L")                  # 3. three channels -> one (grayscale)
    return np.array(img).flatten()          # 4. 2-D grid -> 1-D vector
