import numpy as np
from pathlib import Path
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread, imsave

# Always save/load cache here
ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_DATA_DIR = ROOT_DIR / "examples" / "sample_data"


def load_sample_image(name="cameraman", shape=(512, 512), save_dir: Path = SAMPLE_DATA_DIR) -> np.ndarray:
    """
    Load and normalize a sample image from skimage.data.

    This function always returns an image resized to the specified shape,
    regardless of the cached version's resolution.

    Parameters
    ----------
    name : str
        Name of the image to load from skimage.data (e.g., "camera", "moon").
    shape : tuple
        Desired image shape (height, width).
    save_dir : Path
        Directory to store the original image as PNG (used as cache).

    Returns
    -------
    img : np.ndarray
        Float32 array of shape `shape`, normalized to [0, 1].
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.png"

    if not path.exists():
        # Load from skimage.data
        if not hasattr(data, name):
            raise ValueError(f"Image '{name}' is not available in skimage.data")
        img = getattr(data, name)()
        if img.ndim == 3:
            img = rgb2gray(img)
        img = (img - img.min()) / (img.max() - img.min())
        imsave(path, (img * 255).astype(np.uint8))  # Save normalized 8-bit version

    # Always reload, normalize, and resize
    img = imread(path, as_gray=True).astype(np.float32) / 255.0
    if img.shape != shape:
        img = resize(img, shape, mode="reflect", anti_aliasing=True)

    return img.astype(np.float32)
