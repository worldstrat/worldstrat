import itertools

import kornia
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from pytorch_msssim import ms_ssim

from src.transforms import lanczos_kernel


def tv_loss(x: Tensor) -> Tensor:
    """ Total variation loss.
    The sum of the absolute differences for neighboring pixel-values in the input images.

    Parameters
    ----------
    x : Tensor
        Tensor of shape (batch_size, channels, height, width).

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    height, width = x.shape[-2:]
    return (kornia.losses.total_variation(x) / (height * width)).mean(dim=1)


def ms_ssim_loss(y_hat, y, window_size):
    """ Multi-Scale Structural Similarity loss.
    See: https://www.cns.nyu.edu/pub/eero/wang03b.pdf

    Parameters
    ----------
    y_hat : Tensor
        Output, tensor of shape (batch_size, channels, height, width).
    y : Tensor
        Label, tensor of shape (batch_size, channels, height, width).
    window_size : int
        The size of the gaussian kernel used in the MS-SSIM calculation to smooth the images.

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return 1 - ms_ssim(y_hat, y, data_range=1, win_size=window_size, size_average=False)


def ssim_loss(y_hat, y, window_size=5):
    """ Structural Similarity loss.
    See: http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf

    Parameters
    ----------
    y_hat : Tensor
        Output, tensor of shape (batch_size, channels, height, width).
    y : Tensor
        Label, tensor of shape (batch_size, channels, height, width).
    window_size : int, optional
        The size of the gaussian kernel used in the SSIM calculation to smooth the images, by default 5.

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return kornia.losses.ssim_loss(
        y_hat, y, window_size=window_size, reduction="none"
    ).mean(
        dim=(-1, -2, -3)
    )  # over C, H, W


def mae_loss(y_hat, y):
    """ Mean Absolute Error (L1) loss.
    Sum of all the absolute differences between the label and the output.

    Parameters
    ----------
    y_hat : Tensor
        Output, tensor of shape (batch_size, channels, height, width).
    y : Tensor
        Label, tensor of shape (batch_size, channels, height, width).

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return F.l1_loss(y_hat, y, reduction="none").mean(dim=(-1, -2, -3))  # over C, H, W


def mse_loss(y_hat, y):
    """ Mean Squared Error (L2) loss.
    Sum of all the squared differences between the label and the output.

    Parameters
    ----------
    y_hat : Tensor
        Output, tensor of shape (batch_size, channels, height, width).
    y : Tensor
        Label, tensor of shape (batch_size, channels, height, width).

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return F.mse_loss(y_hat, y, reduction="none").mean(dim=(-1, -2, -3))  # over C, H, W


def psnr_loss(y_hat, y):
    """ Peak Signal to Noise Ratio (PSNR) loss.
    The logarithm of base ten of the mean squared error between the label 
    and the output, multiplied by ten.

    In the proper form, there should be a minus sign in front of the equation, 
    but since we want to maximize the PSNR, 
    we minimize the negative PSNR (loss), thus the leading minus sign has been omitted.

    Parameters
    ----------
    y_hat : Tensor
        Output, tensor of shape (batch_size, channels, height, width).
    y : Tensor
        Label, tensor of shape (batch_size, channels, height, width).

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return 10.0 * torch.log10(mse_loss(y_hat, y))


def get_patch(x, x_start, y_start, patch_size):
    """ Get a patch of the input tensor. 
    The patch begins at the coordinates (x_start, y_start) ,
    ends at (x_start + patch_size, y_start + patch_size).

    Parameters
    ----------
    x : Tensor
        Tensor of shape (batch_size, channels, height, width).
    x_start : int
        The x-coordinate of the top left corner of the patch.
    y_start : int
        The y-coordinate of the top left corner of the patch.
    patch_size : int
        The height/width of the (square) patch.

    Returns
    -------
    Tensor
        Tensor of shape (batch_size, channels, patch_size, patch_size).
    """
    return x[..., x_start : (x_start + patch_size), y_start : (y_start + patch_size)]


class Shift(nn.Module):
    """ A non-learnable convolutional layer for shifting. 
    Used instead of ShiftNet.
    """

    def __init__(self, shift_by_px, mode="discrete", step=1.0, use_cache=True):
        """ Initialize the Shift layer.

        Parameters
        ----------
        shift_by_px : int
            The number of pixels to shift the input by.
        mode : str, optional
            The mode of shifting, by default 'discrete'.
        step : float, optional
            The step size of the shift, by default 1.0.
        use_cache : bool, optional
            Whether to cache the shifts, by default True.
        """
        super().__init__()
        self.shift_by_px = shift_by_px
        self.mode = mode
        if mode == "discrete":
            shift_kernels = self._shift_kernels(shift_by_px)
        elif mode == "lanczos":
            shift_kernels = self._lanczos_kernels(shift_by_px, step)
        self._make_shift_conv2d(shift_kernels)
        self.register_buffer("shift_kernels", shift_kernels)
        self.y = None
        self.y_hat = None
        self.use_cache = use_cache
        self.shift_cache = {}

    def _make_shift_conv2d(self, kernels):
        """ Make the shift convolutional layer.

        Parameters
        ----------
        kernels : torch.Tensor
            The shift kernels.
        """
        self.number_of_kernels, _, self.kernel_height, self.kernel_width = kernels.shape
        self.conv2d_shift = nn.Conv2d(
            in_channels=self.number_of_kernels,
            out_channels=self.number_of_kernels,
            kernel_size=(self.kernel_height, self.kernel_width),
            bias=False,
            groups=self.number_of_kernels,
            padding_mode="reflect",
        )

        # Fix (kN, 1, kH, kW)
        self.conv2d_shift.weight.data = kernels
        self.conv2d_shift.requires_grad_(False)  # Freeze

    @staticmethod
    def _shift_kernels(shift_by_px):
        """ Create the shift kernels.

        Parameters
        ----------
        shift_by_px : int
            The number of pixels to shift the input by.

        Returns
        -------
        torch.Tensor
            The shift kernels.
        """
        kernel_height = kernel_width = (2 * shift_by_px) + 1
        kernels = torch.zeros(
            kernel_height * kernel_width, 1, kernel_height, kernel_width
        )
        all_xy_positions = list(
            itertools.product(range(kernel_height), range(kernel_width))
        )

        for kernel, (x, y) in enumerate(all_xy_positions):
            kernels[kernel, 0, x, y] = 1
        return kernels

    @staticmethod
    def _lanczos_kernels(shift_by_px, shift_step):
        """ Create the Lanczos kernels.

        Parameters
        ----------
        shift_by_px : int
            The number of pixels to shift the input by.
        shift_step : float
            The step size of the shift.

        Returns
        -------
        torch.Tensor
            The Lanczos kernels.
        """
        shift_step = float(shift_step)
        shifts = torch.arange(-shift_by_px, shift_by_px + shift_step, shift_step)
        shifts = shifts[:, None]
        kernel = lanczos_kernel(shifts, kernel_lobes=3)
        kernels = torch.stack(
            [
                kernel_y[:, None] @ kernel_x[None, :]
                for kernel_y, kernel_x in itertools.product(kernel, kernel)
            ]
        )
        return kernels[:, None]

    def forward(self, y: Tensor) -> Tensor:
        """ Forward shift pass.

        Parameters
        ----------
        y : torch.Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The shifted tensor.
        """
        batch_size, input_channels, input_height, input_width = y.shape
        patch_height = patch_width = y.shape[-1] - self.kernel_width + 1

        number_of_kernels_dimension = -3

        # TODO: explain what is going on here
        y = y.unsqueeze(dim=number_of_kernels_dimension).expand(
            -1, -1, self.number_of_kernels, -1, -1
        )  # (B, C, kN, H, W)

        y = y.view(
            batch_size * input_channels,
            self.number_of_kernels,
            input_height,
            input_width,
        )

        # Current input shape: (number_of_kernels, batch_channels, height, width)
        y = self.conv2d_shift(y)
        batch_size_channels, number_of_kernels, height, width = 0, 1, 2, 3

        # Transposed input shape: (number_of_kernels, batch_size_channels, height, width)
        y = y.transpose(number_of_kernels, batch_size_channels)

        y = y.contiguous().view(
            self.number_of_kernels * batch_size,
            input_channels,
            patch_height,
            patch_width,
        )
        return y

    def prep_y_hat(self, y_hat):
        """ Prepare the y_hat for the shift by

        Parameters
        ----------
        y_hat : torch.Tensor
            The output tensor.

        Returns
        -------
        Tensor
            ???
        """
        patch_width = y_hat.shape[-1] - self.kernel_width + 1

        # Get center patch
        y_hat = get_patch(
            y_hat,
            x_start=self.shift_by_px,
            y_start=self.shift_by_px,
            patch_size=patch_width,
        )

        # (number_of_kernels, batch_size, channels, height, width)
        y_hat = y_hat.expand(self.number_of_kernels, -1, -1, -1, -1)
        _, batch_size, channels, height, width = y_hat.shape

        # (number_of_kernels*batch_size, channels, height, width)
        return y_hat.contiguous().view(
            self.number_of_kernels * batch_size, channels, height, width
        )

    @staticmethod
    def gather_shifted_y(y: Tensor, ix) -> Tensor:
        """ Gather the shifted y.

        Parameters
        ----------
        y : Tensor
            The input tensor.
        ix : Tensor
            ???

        Returns
        -------
        Tensor
            The shifted y.
        """
        batch_size = ix.shape[0]
        # TODO: Check if 1st dimension is number of kernels
        number_of_kernels_batch_size, channels, height, width = y.shape
        number_of_kernels = number_of_kernels_batch_size // batch_size
        ix = ix[None, :, None, None, None].expand(-1, -1, channels, height, width)

        # (batch_size, channels, height, width)
        return y.view(number_of_kernels, batch_size, channels, height, width).gather(
            dim=0, index=ix
        )[0]

    @staticmethod
    def _hash_y(y):
        """ Hashes y by [???].

        Parameters
        ----------
        y : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The hashed tensor.
        """
        batch_size = y.shape[0]
        return [tuple(row.tolist()) for row in y[:, :4, :4, :4].reshape(batch_size, -1)]

    def registered_loss(self, loss_function):
        """ Creates a loss function adjusted for registration errors 
        by computing the min loss across shifts of up to `self.shift_by_px` pixels.

        Parameters
        ----------
        loss_function : Callable
            The loss function.

        Returns
        -------
        Callable
            The loss function adjusted for registration errors.

        """

        def _loss(y_hat, y=None, **kws):
            """ Compute the loss.

            Parameters
            ----------
            y_hat : Tensor
                The output tensor.
            y : Tensor, optional
                The target tensor, by default None.

            Returns
            -------
            Tensor
                The loss.
            """

            hashed_y = self._hash_y(y)
            cached_y = (
                torch.Tensor([hash in self.shift_cache for hash in hashed_y])
                .bool()
                .to(y.device)
            )
            not_cached_y = ~cached_y

            # If y and y_hat are both cached, return the loss
            if self.y is not None and self.y_hat is not None:
                min_loss = loss_function(self.y_hat, self.y, **kws)
            else:
                batch_size, channels, height, width = y.shape
                min_loss = torch.zeros(batch_size).to(y.device)
                patch_width = width - self.kernel_width + 1  # patch width

                y_all = torch.zeros(batch_size, channels, patch_width, patch_width).to(
                    y.device
                )

                y_hat_all = torch.zeros(
                    batch_size, channels, patch_width, patch_width
                ).to(y.device)

                # If there are any hashes in cache
                if any(cached_y):

                    ix = torch.stack(
                        [
                            self.shift_cache[hash]
                            for hash in hashed_y
                            if hash in self.shift_cache
                        ]
                    )

                    optimal_shift_kernel = self.shift_kernels[ix]
                    print(optimal_shift_kernel.shape)
                    (
                        batch_size,
                        number_of_kernels,
                        _,
                        kernel_height,
                        kernel_width,
                    ) = optimal_shift_kernel.shape

                    conv2d_shift = nn.Conv2d(
                        in_channels=number_of_kernels,
                        out_channels=number_of_kernels,
                        kernel_size=(kernel_height, kernel_width),
                        bias=False,
                        groups=number_of_kernels,
                        padding_mode="reflect",
                    )

                    # Fix and freeze (kN, 1, kH, kW)
                    conv2d_shift.weight.data = optimal_shift_kernel
                    conv2d_shift.requires_grad_(False)
                    y_in = conv2d_shift(y[cached_y].transpose(-3, -4)).transpose(-4, -3)

                    y_hat_in = get_patch(
                        y_hat[cached_y],
                        x_start=self.shift_by_px,
                        y_start=self.shift_by_px,
                        patch_size=patch_width,
                    )  # center patch

                    min_loss[cached_y] = loss_function(y_hat_in, y_in, **kws)
                    y_all[cached_y] = y_in.to(y_all.dtype)
                    y_hat_all[cached_y] = y_hat_in.to(y_hat_all.dtype)

                # If there are any hashes not in cache
                if any(not_cached_y):
                    y_out, y_hat_out = y[not_cached_y], y_hat[not_cached_y]
                    batch_size = y_out.shape[0]
                    y_out = self(y_out)  # (Nbatch, channels, height, width)
                    # (Nbatch, channels, height, width)
                    y_hat_out = self.prep_y_hat(y_hat_out)
                    losses = loss_function(y_hat_out, y_out, **kws).view(
                        -1, batch_size
                    )  # (N, B)
                    min_loss[not_cached_y], ix = torch.min(
                        losses, dim=0
                    )  # min over patches (B,)
                    y_out = self.gather_shifted_y(
                        y_out, ix
                    )  # shifted y (batch_size, channels, height, width)
                    batch_size, channels, height, width = y_out.shape
                    # (batch_size, channels, height, width). Copied along dim 0
                    y_hat_out = y_hat_out.view(-1, batch_size, channels, height, width)

                    y_hat_out = y_hat_out[0]

                    y_all[not_cached_y] = y_out.to(y_all.dtype)
                    y_hat_all[not_cached_y] = y_hat_out.to(y_hat_all.dtype)
                    if self.use_cache:
                        hashed_y = [
                            hash for hash in hashed_y if hash not in self.shift_cache
                        ]
                        for hash, index in zip(hashed_y, ix):
                            self.shift_cache[hash] = ix

                self.y, self.y_hat = y_all, y_hat_all

            return min_loss

        return _loss
