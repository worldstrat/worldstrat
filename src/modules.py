from typing_extensions import Self
from src.datasources import JIF_S2_MEAN, JIF_S2_STD, S2_ALL_12BANDS
import numpy as np
import torch
import torch.nn.functional as F

from math import log2
from kornia.geometry.transform import Resize
from torch import nn, Tensor
from typing import Tuple, Optional
from src.transforms import Shift, WarpPerspective

###############################################################################
# Layers
###############################################################################


class OneHot(nn.Module):
    """ One-hot encoder. """

    def __init__(self, num_classes):
        """ Initialize the OneHot layer.

        Parameters
        ----------
        num_classes : int
            The number of classes to encode.
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        """ Forward pass. 

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The encoded tensor.
        """
        # Current shape of x: (..., 1, H, W).
        x = x.to(torch.int64)
        # Remove the empty dimension: (..., H, W).
        x = x.squeeze(-3)
        # One-hot encode: (..., H, W, num_classes)
        x = F.one_hot(x, num_classes=self.num_classes)

        # Permute the dimensions so the number of classes is before the height and width.
        if x.ndim == 5:
            x = x.permute(0, 1, 4, 2, 3)  # (..., num_classes, H, W)
        elif x.ndim == 4:
            x = x.permute(0, 3, 1, 2)
        return x


class DoubleConv2d(nn.Module):
    """ Two-layer 2D convolutional block with a PReLU activation in between. """

    # TODO: verify if we still need reflect padding-mode. If we do, set via constructor.

    def __init__(self, in_channels, out_channels, kernel_size=3, use_batchnorm=False):
        """ Initialize the DoubleConv2d layer.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int, optional
            The kernel size, by default 3.
        use_batchnorm : bool, optional
            Whether to use batch normalization, by default False.
        """
        super().__init__()

        self.doubleconv2d = nn.Sequential(
            # ------- First block -------
            # First convolutional layer.
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=not use_batchnorm,
                padding_mode="reflect",
            ),
            # Batch normalization, if requested.
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            # Parametric ReLU activation.
            nn.PReLU(),
            # Dropout regularization, keep probability 0.5.
            nn.Dropout(p=0.5),
            # ------- Second block -------
            # Second convolutional layer.
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=not use_batchnorm,
                padding_mode="reflect",
            ),
            # Batch normalization, if requested.
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            # Parametric ReLU activation.
            nn.PReLU(),
            # Dropout regularization, keep probability 0.5.
            nn.Dropout(p=0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass of the DoubleConv2d layer.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.doubleconv2d(x)


class ResidualBlock(nn.Module):
    """ Two-layer 2D convolutional block (DoubleConv2d) 
    with a skip-connection to a sum."""

    def __init__(self, in_channels, kernel_size=3, **kws):
        """ Initialize the ResidualBlock layer.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        kernel_size : int, optional
            The kernel size, by default 3.
        """
        super().__init__()
        self.residualblock = DoubleConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            **kws,
        )

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass of the ResidualBlock layer.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, in_channels, height, width).
        """
        x = x + self.residualblock(x)
        return x


class DenseBlock(ResidualBlock):
    """ Two-layer 2D convolutional block (DoubleConv2d) with a skip-connection 
    to a concatenation (instead of a sum used in ResidualBlock)."""

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass of the DenseBlock layer.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, in_channels, height, width).
        """
        return torch.cat([x, self.residualblock(x)], dim=1)


class FusionBlock(nn.Module):
    """ A block that fuses two revisits into one. """

    def __init__(self, in_channels, kernel_size=3, use_batchnorm=False):
        """ Initialize the FusionBlock layer.

        Fuse workflow:
        xx ---> xx ---> x
        |       ^
        |-------^

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        kernel_size : int, optional
            The kernel size, by default 3.
        use_batchnorm : bool, optional
            Whether to use batch normalization, by default False.
        """
        super().__init__()

        # TODO: it might be better to fuse the encodings in groups - one per channel.

        number_of_revisits_to_fuse = 2

        self.fuse = nn.Sequential(
            # A two-layer 2D convolutional block with a skip-connection to a sum.
            ResidualBlock(
                number_of_revisits_to_fuse * in_channels,
                kernel_size,
                use_batchnorm=use_batchnorm,
            ),
            # A 2D convolutional layer.
            nn.Conv2d(
                in_channels=number_of_revisits_to_fuse * in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=not use_batchnorm,
                padding_mode="reflect",
            ),
            # Batch normalization, if requested.
            nn.BatchNorm2d(in_channels) if use_batchnorm else nn.Identity(),
            # Parametric ReLU activation.
            nn.PReLU(),
        )

    @staticmethod
    def split(x):
        """ Split the input tensor (revisits) into two parts/halves.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, revisits, in_channels, height, width).

        Returns
        -------
        tuple of Tensor
            The two output tensors of shape (batch_size, revisits//2, in_channels, height, width).
        """

        number_of_revisits = x.shape[1]
        assert number_of_revisits % 2 == 0, f"number_of_revisits={number_of_revisits}"

        # (batch_size, revisits//2, in_channels, height, width)
        first_half = x[:, : number_of_revisits // 2].contiguous()
        second_half = x[:, number_of_revisits // 2 :].contiguous()

        # TODO: return a carry-encoding?
        return first_half, second_half

    def forward(self, x):
        """ Forward pass of the FusionBlock layer.


        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, revisits, in_channels, height, width).
            Revisits encoding of the input.

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, revisits/2, in_channels, height, width).
            Fused encoding of the input.
        """

        first_half, second_half = self.split(x)
        batch_size, half_revisits, channels, height, width = first_half.shape

        first_half = first_half.view(
            batch_size * half_revisits, channels, height, width
        )

        second_half = second_half.view(
            batch_size * half_revisits, channels, height, width
        )

        # Current shape of x: (batch_size * revisits//2, 2*in_channels, height, width)
        x = torch.cat([first_half, second_half], dim=-3)

        # Fused shape of x: (batch_size * revisits//2, in_channels, height, width)
        fused_x = self.fuse(x)

        # Fused encoding shape of x: (batch_size, revisits/2, channels, width, height)
        fused_x = fused_x.view(batch_size, half_revisits, channels, height, width)

        return fused_x


class RecursiveFusion(nn.Module):
    """ Recursively fuses a set of encodings. """

    def __init__(self, in_channels, kernel_size, revisits):
        """ Initialize the RecursiveFusion layer.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        kernel_size : int
            The kernel size.
        revisits : int
            The number of revisits.
        """
        super().__init__()

        log2_revisits = log2(revisits)
        if log2_revisits % 1 == 0:
            num_fusion_layers = int(log2_revisits)
        else:
            num_fusion_layers = int(log2_revisits) + 1

        pairwise_fusion = FusionBlock(in_channels, kernel_size, use_batchnorm=False)

        self.fusion = nn.Sequential(
            *(pairwise_fusion for _ in range(num_fusion_layers))
        )

    @staticmethod
    def pad(x):
        """ Pad the input tensor with black revisits to a power of 2.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, revisits, in_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, revisits, in_channels, height, width).
        """

        # TODO: should we pad with copies of revisits instead of zeros?
        # TODO: move to transforms.py
        batch_size, revisits, channels, height, width = x.shape
        log2_revisits = torch.log2(torch.tensor(revisits))

        if log2_revisits % 1 > 0:

            nextpower = torch.ceil(log2_revisits)
            number_of_black_padding_revisits = int(2 ** nextpower - revisits)

            black_revisits = torch.zeros(
                batch_size,
                number_of_black_padding_revisits,
                channels,
                height,
                width,
                dtype=x.dtype,
                device=x.device,
            )

            x = torch.cat([x, black_revisits], dim=1)
        return x

    def forward(self, x):
        """ Forward pass of the RecursiveFusion layer.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, revisits, in_channels, height, width).

        Returns
        -------
        Tensor
            The fused output tensor of shape (batch_size, in_channels, height, width).
        """
        x = self.pad(x)  # Zero-pad if neccessary to ensure power of 2 revisits
        x = self.fusion(x)  # (batch_size, 1, channels, height, width)
        return x.squeeze(1)  # (batch_size, channels, height, width)


class ConvTransposeBlock(nn.Module):
    """ Upsampler block with ConvTranspose2d. """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        sr_kernel_size,
        zoom_factor,
        use_batchnorm=False,
    ):
        """ Initialize the ConvTransposeBlock layer.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int
            The kernel size.
        sr_kernel_size : int
            The kernel size of the SR convolution.
        zoom_factor : int
            The zoom factor.
        use_batchnorm : bool, optional
            Whether to use batchnorm, by default False.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # TODO: check if sr_kernel_size is the correct name
        self.sr_kernel_size = sr_kernel_size
        self.zoom_factor = zoom_factor
        self.use_batchnorm = use_batchnorm

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=self.kernel_size,
                stride=self.zoom_factor,
                padding=0,
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding="same",
                bias=not use_batchnorm,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(self.in_channels) if use_batchnorm else nn.Identity(),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.sr_kernel_size,
                stride=1,
                padding="same",
                bias=not use_batchnorm,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(self.out_channels) if use_batchnorm else nn.Identity(),
            nn.PReLU(),
        )

    def forward(self, x):
        """ Forward pass of the ConvTransposeBlock layer.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.upsample(x)


class PixelShuffleBlock(ConvTransposeBlock):

    """PixelShuffle block with ConvTranspose2d for sub-pixel convolutions. """

    # TODO: add a Dropout layer between the convolution layers?

    def __init__(self, **kws):
        super().__init__(**kws)
        assert self.in_channels % self.zoom_factor ** 2 == 0
        self.in_channels = self.in_channels // self.zoom_factor ** 2
        self.upsample = nn.Sequential(
            nn.PixelShuffle(self.zoom_factor),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=self.sr_kernel_size,
                stride=1,
                padding="same",
                bias=not self.use_batchnorm,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(self.in_channels) if self.use_batchnorm else nn.Identity(),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.sr_kernel_size,
                stride=1,
                padding="same",
                bias=not self.use_batchnorm,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(self.out_channels) if self.use_batchnorm else nn.Identity(),
            nn.PReLU(),
        )


class HomographyNet(nn.Module):
    """ A network that estimates a parametric geometric transformation matrix
    between two given images.

    Source: https://github.com/mazenmel/Deep-homography-estimation-Pytorch
    Reference: https://arxiv.org/pdf/1606.03798.pdf
    """

    def __init__(self, input_size, in_channels, fc_size, type="translation"):
        """ Initialize the HomographyNet layer.

        Parameters
        ----------
        input_size : tuple of int
            The input size.
        in_channels : int
            The number of input channels.
        fc_size : int
            The number of hidden channels.
        type : str, optional
            The type of transformation, by default 'translation'.

        Raises
        ------
        ValueError
            If the type of transformation is not supported.
        """

        super().__init__()

        if type not in ["translation", "homography"]:
            raise ValueError(
                f"Expected ['translation'|'homography'] for 'type'. Got {type}."
            )
        self.type = type
        self.input_size = input_size
        hidden_channels = fc_size // 2
        kernel_size = 3
        stride = 1
        padding = kernel_size // 2

        if self.kind == "translation":
            n_transform_params = 2
            self.transform = Shift()
        elif self.kind == "homography":
            n_transform_params = 8
            self.transform = WarpPerspective()

        def convblock(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=False,
                    padding_mode="reflect",
                ),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=False,
                    padding_mode="reflect",
                ),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(),
            )

        self.cnn = nn.Sequential(
            convblock(
                in_channels=2 * in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.MaxPool2d(2),
            convblock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.MaxPool2d(2),
            convblock(
                in_channels=hidden_channels,
                out_channels=2 * hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.MaxPool2d(2),
            convblock(
                in_channels=2 * hidden_channels,
                out_channels=2 * hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Sequential(
            nn.Linear(in_features=fc_size, out_features=n_transform_params, bias=False)
        )

    def register(self, image1, image2):
        """ Register two images.

        Parameters
        ----------
        image1 : Tensor
            The first image tensor of shape (batch_size, in_channels, height, width).
        image2 : Tensor
            The second image tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        Tensor
            Parametric transformations on image1 with respect to image2 (batch_size, 2).
        """

        # Center each image by its own mean.
        image1 = image1 - torch.mean(image1, dim=(2, 3), keepdim=True)
        image2 = image2 - torch.mean(image2, dim=(2, 3), keepdim=True)

        # Concat channels (batch_size, 2 * channels, height, width)
        x = torch.cat([image1, image2], dim=1)

        x = self.cnn(x)

        # (batch_size, channels) global average pooling
        x = x.mean(dim=(2, 3))

        x = self.dropout(x)
        transformation_parameters = self.fc(x)
        transformation_parameters = torch.sigmoid(transformation_parameters) * 2 - 1

        return transformation_parameters

    def forward(self, image1, image2):
        """ Forward pass (register and transform two images).

        Parameters
        ----------
        image1 : Tensor
            The first image tensor of shape (batch_size, in_channels, height, width).
        image2 : Tensor
            The second image tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        Tensor
            The transformed image tensor of shape (batch_size, in_channels, height, width).
        """
        with torch.autograd.set_detect_anomaly(True):
            transform_params = self.register(image1, image2)
            return self.transform(image1, transform_params)


###############################################################################
# Models
###############################################################################


class SRCNN(nn.Module):
    """ Super-resolution CNN.
    Uses no recursive function, revisits are treated as channels.
    """

    def __init__(
        self,
        in_channels,
        mask_channels,
        revisits,
        hidden_channels,
        out_channels,
        kernel_size,
        residual_layers,
        output_size,
        zoom_factor,
        sr_kernel_size,
        registration_kind,
        homography_fc_size,
        use_reference_frame=False,
        **kws,
    ) -> None:
        """ Initialize the Super-resolution CNN.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        mask_channels : int
            The number of mask channels.
        revisits : int
            The number of revisits.
        hidden_channels : int
            The number of hidden channels.
        out_channels : int
            The number of output channels.
        kernel_size : int
            The kernel size.
        residual_layers : int
            The number of residual layers.
        output_size : tuple of int
            The output size.
        zoom_factor : int
            The zoom factor.
        sr_kernel_size : int
            The kernel size of the SR convolution.
        registration_kind : str, optional
            The kind of registration.
        homography_fc_size : int
            The size of the fully connected layer for the homography.
        use_reference_frame : bool, optional
            Whether to use the reference frame, by default False.
        """
        super().__init__()

        self.in_channels = 2 * in_channels if use_reference_frame else in_channels
        self.mask_channels = mask_channels
        self.revisits = revisits
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.residual_layers = residual_layers
        self.output_size = output_size
        self.zoom_factor = zoom_factor
        self.sr_kernel_size = sr_kernel_size
        self.registration_kind = registration_kind
        self.homography_fc_size = homography_fc_size
        self.use_batchnorm = False
        self.use_reference_frame = use_reference_frame

        # Image encoder
        self.encoder = DoubleConv2d(
            in_channels=self.in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            use_batchnorm=self.use_batchnorm,
        )

        # Mask encoder
        self.mask_encoder = nn.Sequential(
            OneHot(num_classes=12),
            DoubleConv2d(in_channels=self.mask_channels, out_channels=1, kernel_size=3),
            nn.Sigmoid(),
        )

        # TODO: does it make sense to reuse the same Res-Block object in nn.Sequential?
        # Fusion
        self.doubleconv2d = DoubleConv2d(
            in_channels=hidden_channels * revisits,  # revisits as channels
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            use_batchnorm=self.use_batchnorm,
        )
        self.residualblocks = nn.Sequential(
            *(
                ResidualBlock(
                    in_channels=hidden_channels,
                    kernel_size=kernel_size,
                    use_batchnorm=self.use_batchnorm,
                )
                for _ in range(residual_layers)
            )
        )
        self.fusion = nn.Sequential(self.doubleconv2d, self.residualblocks)

        ## Super-resolver (upsampler + renderer)
        self.sr = PixelShuffleBlock(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            sr_kernel_size=sr_kernel_size,
            zoom_factor=zoom_factor,
            use_batchnorm=self.use_batchnorm,
        )

        self.resize = Resize(
            self.output_size,
            interpolation="bilinear",
            align_corners=False,
            antialias=True,
        )

    # TODO: cleanest as reference frame
    # TODO: filter for pad-revisits before taking the median
    def reference_frame(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """ Compute the reference frame as the median of all low-res revisits.

        Parameters
        ----------
        x : Tensor
            The input tensor (low-res revisits).
        mask : Tensor, optional
            The mask tensor of shape.

        Returns
        -------
        Tensor
            The reference frame.
        """
        return x.median(dim=-4, keepdim=True).values

    def forward(
        self, x: Tensor, y: Optional[Tensor] = None, mask: Optional[Tensor] = None,
    ) -> Tensor:
        """ Forward pass of the Super-resolution CNN.

        Parameters
        ----------
        x : Tensor
            The input tensor (low-res revisits).
            Shape: (batch_size, revisits, channels, height, width).
        y : Tensor, optional
            The target tensor (high-res revisits).
            Shape: (batch_size, 1, channels, height, width).
        mask : Tensor, optional
            The mask tensor.y
            Shape: (batch_size, revisits, mask_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor (super-resolved images).
            Shape: (batch_size, 1, channels, height, width).
        """
        if self.use_reference_frame:
            # Concatenated shape: (batch_size, revisits, 2*channels, height, width)
            x = self.compute_and_concat_reference_frame_to_input(x)

        batch_size, revisits, channels, height, width = x.shape
        hidden_channels = self.hidden_channels

        x = x.view(batch_size * revisits, channels, height, width)

        # Encoded shape: (batch_size * revisits, hidden_channels, height, width)
        x = self.encoder(x)

        # Concatenated shape:
        # (batch_size * revisits, hidden_channels+mask_channels, height, width)
        x, mask_channels = self.encode_and_concatenate_masks_to_input(
            x, mask, batch_size, revisits, height, width
        )

        x = x.view(
            batch_size, revisits * (hidden_channels + mask_channels), height, width
        )
        # Fused shape: (batch_size, hidden_channels, height, width)
        x = self.fusion(x)
        x = self.sr(x)
        # Ensure output size of (batch_size, channels, height, width)
        x = self.resize(x)
        # Pad with empty revisit dimension: (batch_size, 1, channels, height, width)
        x = x[:, None]
        return x

    def compute_and_concat_reference_frame_to_input(self, x):
        # Current shape: (batch_size, revisits, channels, height, width)
        reference_frame = self.reference_frame(x).expand_as(x)
        # Concatenated shape: (batch_size, revisits, 2*channels, height, width)
        x = torch.cat([x, reference_frame], dim=-3)
        return x

    def encode_and_concatenate_masks_to_input(
        self, x, mask, batch_size, revisits, height, width
    ):
        if mask is not None:
            mask, mask_channels = mask, self.mask_channels
            mask = mask.view(batch_size * revisits, mask_channels, height, width)
            # Encoded shape: (batch_size * revisits, mask_channels, height, width)
            mask = self.mask_encoder(mask)
            mask_channels = mask.shape[-3]
            # Concatenated shape:
            # (batch_size * revisits, hidden_channels+mask_channels, height, width)
            x = torch.cat([x, mask], dim=-3)
        else:
            mask_channels = 0
        return x, mask_channels


class HighResNet(SRCNN):
    """ High-resolution CNN.
    Inherits as many elements from SRCNN as possible for as fair a comparison:
    - DoubleConv2d: the in_channels are doubled by use_reference_frame in SRCNN init
    - Encoder: nn.Sequential(DoubleConv2d, ResidualBlock)
    """

    def __init__(self, skip_paddings=True, **kws) -> None:
        super().__init__(**kws)

        self.skip_paddings = skip_paddings
        self.fusion = RecursiveFusion(
            in_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            # skip_paddings=skip_paddings,
            revisits=self.revisits,
        )

    def forward(self, x, y=None, mask=None):
        """ Forward pass of the High-resolution CNN.

        Parameters
        ----------
        x : Tensor
            The input tensor (low-res revisits).
            Shape: (batch_size, revisits, channels, height, width).
        y : Tensor, optional
            The target tensor (high-res revisits).
            Shape: (batch_size, 1, channels, height, width).
        mask : Tensor, optional
            The mask tensor.
            Shape: (batch_size, revisits, mask_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor (super-resolved images).
            Shape: (batch_size, 1, channels, height, width).
        """
        hidden_channels = self.hidden_channels

        if self.use_reference_frame:
            x = self.compute_and_concat_reference_frame_to_input(x)

        # Note: we could put all these layers in a Sequential, but we
        # would lose the logging abilities for inspection and lose
        # on readability.
        batch_size, revisits, channels, height, width = x.shape
        x = x.view(batch_size * revisits, channels, height, width)
        # Encoded shape: (batch_size * revisits, hidden_channels, height, width)
        x = self.encoder(x)

        x, mask_channels = self.encode_and_concatenate_masks_to_input(
            x, mask, batch_size, revisits, height, width
        )

        x = x.view(batch_size, revisits, hidden_channels + mask_channels, height, width)
        # Fused shape: (batch_size, hidden_channels, height, width)
        x = self.fusion(x)
        # Super-resolved shape:
        # (batch_size, out_channels, height * zoom_factor, width * zoom_factor)
        x = self.sr(x)
        # Ensure output size of (batch_size, channels, height, width)
        x = self.resize(x)

        # Pad with empty revisit dimension: (batch_size, 1, channels, height, width)
        x = x[:, None]
        return x


class BicubicUpscaledBaseline(nn.Module):
    """ Bicubic upscaled single-image baseline. """

    def __init__(
        self, input_size, output_size, chip_size, interpolation="bicubic", device=None, **kws
    ):
        """ Initialize the BicubicUpscaledBaseline.

        Parameters
        ----------
        input_size : tuple of int
            The input size.
        output_size : tuple of int
            The output size.
        chip_size : tuple of int
            The chip size.
        interpolation : str, optional
            The interpolation method, by default 'bicubic'.
            Available methods: 'nearest', 'bilinear', 'bicubic'.
        """
        super().__init__()
        assert interpolation in ["bilinear", "bicubic", "nearest"]
        self.resize = Resize(output_size, interpolation=interpolation)
        self.output_size = output_size
        self.input_size = input_size
        self.chip_size = chip_size
        self.lr_bands = np.array(S2_ALL_12BANDS["true_color"]) - 1
        self.mean = JIF_S2_MEAN[self.lr_bands]
        self.std = JIF_S2_STD[self.lr_bands]
        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass of the BicubicUpscaledBaseline.

        Parameters
        ----------
        x : Tensor
            The input tensor (a batch of low-res revisits).
            Shape: (batch_size, revisits, channels, height, width).

        Returns
        -------
        Tensor
            The output tensor (a single upscaled low-res revisit).
        """
        # If all bands are used, get only the RGB bands for WandB image logging
        if x.shape[2] > 3:
            x = x[:, :, S2_ALL_12BANDS["true_color_zero_index"]]
        # Select the first revisit
        x = x[:, 0, :]

        # Pad with empty revisit dimension: (batch_size, 1, channels, height, width)
        x = x[:, None, :]

        # Normalisation on the channel axis:
        # Add the mean and multiply by the standard deviation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if self.device is None else self.device
        x += torch.as_tensor(self.mean[None, None, ..., None, None]).to(device)
        x *= torch.as_tensor(self.std[None, None, ..., None, None]).to(device)

        # Convert to float, and scale to [0, 1]:
        x = x.float()
        x /= torch.max(x)
        torch.clamp_(x, 0, 1)

        # Upscale to the output size:
        x = self.resize(x)  # upscale (..., T, C, H_o, W_o)
        return x
