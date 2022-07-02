import matplotlib.pyplot as plt
import torch
from torch import Tensor


def showtensor(tensor: Tensor, figsize: int = 4, axis: bool = True, title=None) -> None:

    """ Show a tensor as an image using matplotlib.

    Parameters
    ----------
    tensor : Tensor
        The tensor to show.
    figsize : int, optional
        The size of the figure. The default is 4.
    axis : bool, optional
        Whether to show the axis. The default is True.
    """

    tensor = ensure_tensor_has_four_dimensions(tensor)
    images_or_revisits, channels, height, width = tensor.shape
    nrows, ncols = 1, images_or_revisits
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * figsize, nrows * figsize),
        tight_layout=True,
        squeeze=False,
    )
    
    if title is not None:
        plt.title(title)

    for image in range(images_or_revisits):

        x = tensor[image]

        # Convert images to float
        if x.dtype == torch.uint8:
            x = x / 255.0
        if x.is_floating_point():
            if x.ndim == 3 and x.shape[0] >= 3:
                # Channels first to channels last [1,2,3] -> [3,2,1]
                x = x.permute(1, 2, 0)
                ax[0, image].imshow(x, vmin=0, vmax=1, interpolation="nearest")
            else:
                x = x[0]
                ax[0, image].imshow(x)

        ax[0, image].axis(axis if image == 0 else False)


def ensure_tensor_has_four_dimensions(tensor):
    """ Ensure that the tensor has four dimensions.
    If it doesn't pad the missing dimensions with empty dimensions.

    Parameters
    ----------
    tensor : Tensor
        The tensor to pad.

    Returns
    -------
    Tensor
        A tensor with four dimensions.
    """
    if tensor.ndim == 4:
        pass
    elif tensor.ndim == 3:
        tensor = tensor[None]
    elif tensor.ndim == 2:
        tensor = tensor[None, None]
    return tensor
