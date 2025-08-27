import numpy as np
from pathlib import Path
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread, imsave


def load_sample_image(name="cameraman", shape=(512, 512), save_dir=Path("examples/sample_data")) -> np.ndarray:
    """
    Load and normalize a sample image from skimage.data.

    Parameters
    ----------
    name : str
        The name of the image to load from skimage.data (e.g., "cameraman", "moon", "eagle").
    shape : tuple
        The shape (height, width) to resize the image to.
    save_dir : Path
        Directory to cache the image as a .png file.

    Returns
    -------
    img : np.ndarray
        A float32 NumPy array of shape `shape`, normalized to [0, 1].
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.png"

    if path.exists():
        img = imread(path, as_gray=True)
    else:
        # Load from skimage.data
        if not hasattr(data, name):
            raise ValueError(f"Image '{name}' is not available in skimage.data")
        img = getattr(data, name)()

        # Convert to grayscale if RGB
        if img.ndim == 3:
            img = rgb2gray(img)

        # Resize and normalize
        img = resize(img, shape, mode="reflect", anti_aliasing=True)
        img = (img - img.min()) / (img.max() - img.min())

        # Cache as PNG
        imsave(path, img)

    return img.astype(np.float32)
