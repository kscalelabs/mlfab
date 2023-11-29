# flake8: noqa: E501
"""Defines custom loss functions."""

import functools
import math
import warnings
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def log_cosh_loss(pred: Tensor, target: Tensor) -> Tensor:
    loss = pred - target
    return torch.log(torch.cosh(loss))


def pseudo_huber_loss(
    x: Tensor,
    y: Tensor,
    dim: int = -1,
    factor: float = 0.00054,
    keepdim: bool = False,
) -> Tensor:
    """Returns the pseudo-Huber loss.

    This is taken from the Consistency Models paper.

    Args:
        x: The input tensor.
        y: The target tensor.
        dim: The dimension to compute the loss over.
        factor: The factor to use in the loss.
        keepdim: Whether to keep the dimension or not.

    Returns:
        The pseudo-Huber loss over the given dimension (i.e., that )
    """
    c = factor * math.sqrt(x.shape[dim])
    return torch.sqrt(torch.norm(x - y, p=2, dim=dim, keepdim=keepdim) ** 2 + c**2) - c


def kl_div_single_loss(mu: Tensor, log_var: Tensor, *, clamp_min: float = -30.0, clamp_max: float = 20.0) -> Tensor:
    r"""Computes the KL-divergence loss for a single Gaussian distribution.

    This loss minimizes the KL-divergence between the given distribution and a
    standard normal distribution. This can be expressed as:

    .. math::

        \frac{1}{2} \sum_{i=1}^d \left( \mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1 \right)

    One way of interpretting KL-divergence used here is as the amount of
    information lost when the standard normal distribution is used to
    approximate the given distribution. In other words, by minimizing this loss,
    we are trying to make the given distribution have the same amount of
    information as the standard normal distribution. This is useful for
    things like variational autoencoders, where we want to make the latent
    distribution as close to a standard normal distribution as possible,
    so that we can sample from the normal distribution.

    Args:
        mu: The mean of the Gaussian distribution.
        log_var: The log variance of the Gaussian distribution.
        clamp_min: The minimum value to clamp the log variance to.
        clamp_max: The maximum value to clamp the log variance to.

    Returns:
        The KL-divergence loss.
    """
    log_var = log_var.clamp(min=clamp_min, max=clamp_max)
    var = log_var.exp()
    return -0.5 * (1 + log_var - mu.pow(2) - var)


def kl_div_pair_loss(
    mu_p: Tensor,
    log_var_p: Tensor,
    mu_q: Tensor,
    log_var_q: Tensor,
    *,
    clamp_min: float = -30.0,
    clamp_max: float = 20.0,
) -> Tensor:
    r"""Computes the KL-divergence loss for a pair of Gaussian distributions.

    This loss minimizes the KL-divergence between the first distribution and the
    second distribution. This can be expressed as:

    .. math::

        D_{KL}(p || q) = \sum_{i=1}^d \log \left( \frac{\sigma_{q,i}^2}{\sigma_{p,i}^2} \right) + \frac{\sigma_{p,i}^2 + (\mu_{p,i} - \mu_{q,i})^2}{\sigma_{q,i}^2} - \frac{1}{2}

    One way of interpretting KL-divergence is as the amount of information lost
    when the second distribution is used to approximate the first distribution.
    Thus, the loss is not symmetric.

    Args:
        mu_p: The mean of the first Gaussian distribution.
        log_var_p: The log variance of the first Gaussian distribution.
        mu_q: The mean of the second Gaussian distribution.
        log_var_q: The log variance of the second Gaussian distribution.
        clamp_min: The minimum value to clamp the log variance to.
        clamp_max: The maximum value to clamp the log variance to.

    Returns:
        The KL-divergence loss.
    """
    log_var_p = log_var_p.clamp(min=clamp_min, max=clamp_max)
    log_var_q = log_var_q.clamp(min=clamp_min, max=clamp_max)
    var1 = log_var_p.exp()
    var2 = log_var_q.exp()
    return (log_var_q - log_var_p) + (var1 + (mu_p - mu_q).pow(2)) / var2 - 0.5


SsimFn = Literal["avg", "std"]


class SSIMLoss(nn.Module):
    """Computes structural similarity loss (SSIM).

    The `dynamic_range` is the difference between the maximum and minimum
    possible values for the image. This value is the actually the negative
    SSIM, so that minimizing it maximizes the SSIM score.

    Parameters:
        kernel_size: Size of the Gaussian kernel.
        stride: Stride of the Gaussian kernel.
        channels: Number of channels in the image.
        mode: Mode of the SSIM function, either ``avg`` or ``std``. The
            ``avg`` mode uses unweighted ``(K, K)`` regions, while the ``std``
            mode uses Gaussian weighted ``(K, K)`` regions, which allows for
            larger regions without worrying about blurring.
        sigma: Standard deviation of the Gaussian kernel.
        dynamic_range: Difference between the maximum and minimum possible
            values for the image.

    Inputs:
        x: float tensor with shape ``(B, C, H, W)``
        y: float tensor with shape ``(B, C, H, W)``

    Outputs:
        float tensor with shape ``(B, C, H - K + 1, W - K + 1)``
    """

    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 1,
        channels: int = 3,
        mode: SsimFn = "avg",
        sigma: float = 1.0,
        dynamic_range: float = 1.0,
    ) -> None:
        super().__init__()

        self.c1 = (0.01 * dynamic_range) ** 2
        self.c2 = (0.03 * dynamic_range) ** 2

        match mode:
            case "avg":
                window = self.get_avg_window(kernel_size)
            case "std":
                window = self.get_gaussian_window(kernel_size, sigma)
            case _:
                raise NotImplementedError(f"Unexpected mode: {mode}")

        window = window.expand(channels, 1, kernel_size, kernel_size)
        self.window = nn.Parameter(window.clone(), requires_grad=False)
        self.stride = stride

    def get_gaussian_window(self, ksz: int, sigma: float) -> Tensor:
        x = torch.linspace(-(ksz // 2), ksz // 2, ksz)
        num = (-0.5 * (x / float(sigma)) ** 2).exp()
        denom = sigma * math.sqrt(2 * math.pi)
        window_1d = num / denom
        return window_1d[:, None] * window_1d[None, :]

    def get_avg_window(self, ksz: int) -> Tensor:
        return torch.full((ksz, ksz), 1 / (ksz**2))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.flatten(0, -4)
        y = y.flatten(0, -4)

        channels = x.size(1)
        mu_x = F.conv2d(x, self.window, groups=channels, stride=self.stride)
        mu_y = F.conv2d(y, self.window, groups=channels, stride=self.stride)
        mu_x_sq, mu_y_sq, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y

        sigma_x = F.conv2d(x**2, self.window, groups=channels, stride=self.stride) - mu_x_sq
        sigma_y = F.conv2d(y**2, self.window, groups=channels, stride=self.stride) - mu_y_sq
        sigma_xy = F.conv2d(x * y, self.window, groups=channels, stride=self.stride) - mu_xy

        num_a = 2 * mu_x * mu_y + self.c1
        num_b = 2 * sigma_xy + self.c2
        denom_a = mu_x_sq + mu_y_sq + self.c1
        denom_b = sigma_x**2 + sigma_y**2 + self.c2

        score = (num_a * num_b) / (denom_a * denom_b)
        return -score


class ImageGradLoss(nn.Module):
    """Computes image gradients, for smoothing.

    This function convolves the image with a special Gaussian kernel that
    contrasts the current pixel with the surrounding pixels, such that the
    output is zero if the current pixel is the same as the surrounding pixels,
    and is larger if the current pixel is different from the surrounding pixels.

    Parameters:
        kernel_size: Size of the Gaussian kernel.
        sigma: Standard deviation of the Gaussian kernel.

    Inputs:
        x: float tensor with shape ``(B, C, H, W)``

    Outputs:
        float tensor with shape ``(B, C, H - ksz + 1, W - ksz + 1)``
    """

    kernel: Tensor

    def __init__(self, kernel_size: int = 3, sigma: float = 1.0) -> None:
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd"
        assert kernel_size > 1, "Kernel size must be greater than 1"

        self.kernel_size = kernel_size
        self.register_buffer("kernel", self.get_kernel(kernel_size, sigma), persistent=False)

    def get_kernel(self, ksz: int, sigma: float) -> Tensor:
        x = torch.linspace(-(ksz // 2), ksz // 2, ksz)
        num = (-0.5 * (x / float(sigma)) ** 2).exp()
        denom = sigma * math.sqrt(2 * math.pi)
        window_1d = num / denom
        window = window_1d[:, None] * window_1d[None, :]
        window[ksz // 2, ksz // 2] = 0
        window = window / window.sum()
        window[ksz // 2, ksz // 2] = -1.0
        return window.unsqueeze(0).unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        channels = x.size(1)
        return F.conv2d(x, self.kernel.repeat_interleave(channels, 0), stride=1, padding=0, groups=channels)


class _Scale(nn.Module):
    def __init__(
        self,
        shift: tuple[float, float, float] = (-0.030, -0.088, -0.188),
        scale: tuple[float, float, float] = (0.458, 0.488, 0.450),
    ) -> None:
        super().__init__()

        self.register_buffer("shift", torch.tensor(shift, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)

    shift: Tensor
    scale: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale + self.shift


class _VGG16(nn.Module):
    def __init__(self, pretrained: bool = True, requires_grad: bool = False) -> None:
        super().__init__()

        try:
            import torchvision
        except ModuleNotFoundError:
            raise ModuleNotFoundError("This module requires torchvision: `pip install torchvision`")

        features = torchvision.models.vgg16(pretrained=pretrained).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3


class LPIPS(nn.Module):
    """Computes the learned perceptual image patch similarity (LPIPS) loss.

    This function extracts the VGG-16 features from each input image, projects
    them once, then computes the L2 distance between the projected features.

    The input images should be in the range ``[0, 1]``. The height and width of
    the input images should be at least 64 pixels but can otherwise be
    arbitrary.

    Parameters:
        pretrained: Whether to use the pretrained VGG-16 model. This should
            usually only be disabled for testing.
        requires_grad: Whether to require gradients for the VGG-16 model. This
            should usually be disabled, unless you want to fine-tune the model.
        dropout: Dropout probability for the input projections.

    Inputs:
        image_a: float tensor with shape ``(B, C, H, W)``
        image_b: float tensor with shape ``(B, C, H, W)``

    Outputs:
        float tensor with shape ``(B,)``
    """

    def __init__(self, pretrained: bool = True, requires_grad: bool = False, dropout: float = 0.5) -> None:
        super().__init__()

        # Loads the VGG16 model.
        self.vgg16 = _VGG16(pretrained=pretrained, requires_grad=requires_grad)

        # Scaling layer.
        self.scale = _Scale()

        # Input projections.
        self.in_projs = cast(
            list[nn.Conv2d],
            nn.ModuleList(
                [
                    self._in_proj(64, dropout=dropout),
                    self._in_proj(128, dropout=dropout),
                    self._in_proj(256, dropout=dropout),
                    self._in_proj(512, dropout=dropout),
                    self._in_proj(512, dropout=dropout),
                ]
            ),
        )

    def _in_proj(self, in_channels: int, out_channels: int = 1, dropout: float = 0.5) -> nn.Module:
        if dropout > 0:
            return nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
            )
        return nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def _normalize(self, x: Tensor, eps: float = 1e-10) -> Tensor:
        norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
        return x / (norm_factor + eps)

    def forward(self, image_a: Tensor, image_b: Tensor) -> Tensor:
        image_a, image_b = self.scale(image_a), self.scale(image_b)

        h0_a, h1_a, h2_a, h3_a, h4_a = self.vgg16(image_a)
        h0_b, h1_b, h2_b, h3_b, h4_b = self.vgg16(image_b)

        losses: list[Tensor] = []
        for in_proj, (a, b) in zip(
            self.in_projs,
            ((h0_a, h0_b), (h1_a, h1_b), (h2_a, h2_b), (h3_a, h3_b), (h4_a, h4_b)),
        ):
            diff = (self._normalize(a) - self._normalize(b)).pow(2)
            losses.append(in_proj.forward(diff).mean(dim=(2, 3)))

        return torch.stack(losses, dim=-1).sum(dim=-1).squeeze(1)


WindowFn = Literal["hann", "hamming", "blackman"]


def get_stft_window(window: WindowFn, win_length: int) -> Tensor:
    """Gets a window tensor from a function name.

    Args:
        window: The window function name.
        win_length: The window length.

    Returns:
        The window tensor, with shape ``(win_length)``.
    """
    match window:
        case "hann":
            return torch.hann_window(win_length)
        case "hamming":
            return torch.hamming_window(win_length)
        case "blackman":
            return torch.blackman_window(win_length)
        case _:
            raise NotImplementedError(f"Unexpected window type: {window}")


def stft(x: Tensor, fft_size: int, hop_size: int, win_length: int, window: Tensor) -> Tensor:
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x: Input signal tensor with shape ``(B, T)``.
        fft_size: FFT size.
        hop_size: Hop size.
        win_length: Window length.
        window: The window function.

    Returns:
        Magnitude spectrogram with shape ``(B, num_frames, fft_size // 2 + 1)``.
    """
    dtype = x.dtype
    if dtype == torch.bfloat16:
        x = x.float()
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
    real, imag = x_stft.real, x_stft.imag
    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1).to(dtype)


def spectral_convergence_loss(x_mag: Tensor, y_mag: Tensor, eps: float = 1e-8) -> Tensor:
    """Spectral convergence loss module.

    Args:
        x_mag: Magnitude spectrogram of predicted signal, with shape
            ``(B, num_frames, #=num_freq_bins)``.
        y_mag: Magnitude spectrogram of groundtruth signal, with shape
            ``(B, num_frames, num_freq_bins)``.
        eps: A small value to avoid division by zero.

    Returns:
        Spectral convergence loss value.
    """
    x_mag, y_mag = x_mag.float(), y_mag.float().clamp_min(eps)
    if y_mag.requires_grad:
        warnings.warn(
            "`y_mag` is the ground truth and should not require a gradient! "
            "`spectral_convergence_loss` is not a symmetric loss function."
        )
    return torch.norm(y_mag - x_mag, p="fro", dim=-1) / torch.norm(y_mag, p="fro", dim=-1)


def log_stft_magnitude_loss(x_mag: Tensor, y_mag: Tensor, eps: float = 1e-8) -> Tensor:
    """Log STFT magnitude loss module.

    Args:
        x_mag: Magnitude spectrogram of predicted signal
            ``(B, num_frames, num_freq_bins)``.
        y_mag: Magnitude spectrogram of groundtruth signal
            ``(B, num_frames, num_freq_bins)``.
        eps: A small value to avoid log(0).

    Returns:
        Tensor: Log STFT magnitude loss value.
    """
    x_mag, y_mag = x_mag.float().clamp_min(eps), y_mag.float().clamp_min(eps)
    return F.l1_loss(torch.log(y_mag), torch.log(x_mag), reduction="none").mean(-1)


def stft_magnitude_loss(x_mag: Tensor, y_mag: Tensor) -> Tensor:
    """STFT magnitude loss module.

    Args:
        x_mag: Magnitude spectrogram of predicted signal
            ``(B, num_frames, num_freq_bins)``.
        y_mag: Magnitude spectrogram of groundtruth signal
            ``(B, num_frames, num_freq_bins)``.

    Returns:
        Tensor: STFT magnitude loss value.
    """
    return F.l1_loss(y_mag, x_mag, reduction="none").mean(-1)


class STFTLoss(nn.Module):
    r"""Defines a STFT loss function.

    This function returns two losses which are roughly equivalent, one for
    minimizing the spectral distance and one for minimizing the log STFT
    magnitude distance. The spectral convergence loss is defined as:

    .. math::

        L_{spec} = \\frac{\\|Y - X\\|_F}{\\|Y\\|_F}

    where :math:`X` and :math:`Y` are the predicted and groundtruth STFT
    spectrograms, respectively. The log STFT magnitude loss is defined as:

    .. math::

        L_{mag} = \\frac{\\|\\log Y - \\log X\\|_1}{N}

    Parameters:
        fft_size: FFT size, meaning the number of Fourier bins.
        shift_size: Shift size in sample.
        win_length: Window length in sample.
        window: Window function type. Choices are ``hann``, ``hamming`` and
            ``blackman``.

    Inputs:
        x: Predicted signal ``(B, T)``.
        y: Groundtruth signal ``(B, T)``.

    Outputs:
        Spectral convergence loss value and log STFT magnitude loss value.
    """

    window: Tensor

    def __init__(
        self,
        fft_size: int = 1024,
        shift_size: int = 120,
        win_length: int = 600,
        window: WindowFn = "hann",
    ) -> None:
        super().__init__()

        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", get_stft_window(window, win_length), persistent=False)

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = spectral_convergence_loss(x_mag, y_mag)
        mag_loss = log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi resolution STFT loss module.

    Parameters:
        fft_sizes: List of FFT sizes.
        hop_sizes: List of hop sizes.
        win_lengths: List of window lengths.
        window: Window function type. Choices are ``hann``, ``hamming`` and
            ``blackman``.
        factor_sc: A balancing factor across different losses.
        factor_mag: A balancing factor across different losses.

    Inputs:
        x: Predicted signal (B, T).
        y: Groundtruth signal (B, T).

    Outputs:
        Multi resolution spectral convergence loss value, and multi resolution
        log STFT magnitude loss value.
    """

    def __init__(
        self,
        fft_sizes: list[int] = [1024, 2048, 512],
        hop_sizes: list[int] = [120, 240, 60],
        win_lengths: list[int] = [600, 1200, 300],
        window: WindowFn = "hann",
        factor_sc: float = 1.0,
        factor_mag: float = 1.0,
    ) -> None:
        super().__init__()

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        assert len(fft_sizes) > 0

        self.stft_losses = cast(list[STFTLoss], nn.ModuleList())
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        sc_loss: Tensor | None = None
        mag_loss: Tensor | None = None

        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_l, mag_l = sc_l.flatten(1).mean(1), mag_l.flatten(1).mean(1)
            sc_loss = sc_l if sc_loss is None else sc_loss + sc_l
            mag_loss = mag_l if mag_loss is None else mag_loss + mag_l

        assert sc_loss is not None
        assert mag_loss is not None

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc * sc_loss, self.factor_mag * mag_loss


class MelLoss(nn.Module):
    """Defines a Mel loss function.

    This module is similar to ``STFTLoss``, but it uses mel spectrogram instead
    of the regular STFT, which may be more suitable for speech.

    Parameters:
        sample_rate: Sample rate of the input signal.
        n_fft: FFT size.
        win_length: Window length.
        hop_length: Hop size.
        f_min: Minimum frequency.
        f_max: Maximum frequency.
        n_mels: Number of mel bins.
        window: Window function name.
        power: Exponent for the magnitude spectrogram.
        normalized: Whether to normalize by number of frames.

    Inputs:
        x: Predicted signal ``(B, T)``.
        y: Groundtruth signal ``(B, T)``.

    Outputs:
        Spectral convergence loss value and log mel spectrogram loss value.
    """

    __constants__ = ["eps"]

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 400,
        win_length: int | None = None,
        hop_length: int | None = None,
        f_min: float = 0.0,
        f_max: float | None = None,
        n_mels: int = 80,
        window: WindowFn = "hann",
        power: float = 1.0,
        normalized: bool = False,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()

        try:
            from torchaudio.transforms import MelSpectrogram
        except ModuleNotFoundError:
            raise ModuleNotFoundError("This module requires torchaudio: `pip install torchaudio`")

        self.mel_fn = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            window_fn=functools.partial(get_stft_window, window),
            power=power,
            normalized=normalized,
        )

        self.eps = eps

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        x_mag, y_mag = self.mel_fn(x), self.mel_fn(y)
        return spectral_convergence_loss(x_mag, y_mag, self.eps), log_stft_magnitude_loss(x_mag, y_mag, self.eps)


class MFCCLoss(nn.Module):
    """Defines an MFCC loss function.

    This is similar to ``MelLoss``, but it uses MFCC instead of mel spectrogram.
    MFCCs are like the "spectrum of a spectrum" which are usually just used to
    compress the representation. In the context of a loss function it should
    be largely equivalent to the mel spectrogram, although it may be more
    robust to noise.

    Parameters:
        sample_rate: Sample rate of the input signal.
        n_mfcc: Number of MFCCs.
        dct_type: DCT type.
        norm: Norm type.
        log_mels: Whether to use log-mel spectrograms instead of mel.
        n_fft: FFT size, for Mel spectrogram.
        win_length: Window length, for Mel spectrogram.
        hop_length: Hop size, for Mel spectrogram.
        f_min: Minimum frequency, for Mel spectrogram.
        f_max: Maximum frequency, for Mel spectrogram.
        n_mels: Number of mel bins, for Mel spectrogram.
        window: Window function name, for Mel spectrogram.

    Inputs:
        x: Predicted signal ``(B, T)``.
        y: Groundtruth signal ``(B, T)``.

    Outputs:
        Spectral convergence loss value and log mel spectrogram loss value.
    """

    def __init__(
        self,
        sample_rate: int,
        n_mfcc: int = 40,
        dct_type: int = 2,
        norm: str | None = "ortho",
        log_mels: bool = False,
        n_fft: int = 400,
        win_length: int | None = None,
        hop_length: int | None = None,
        f_min: float = 0.0,
        f_max: float | None = None,
        n_mels: int = 80,
        window: WindowFn = "hann",
    ) -> None:
        super().__init__()

        try:
            from torchaudio.transforms import MFCC
        except ModuleNotFoundError:
            raise ModuleNotFoundError("This module requires torchaudio: `pip install torchaudio`")

        self.mfcc_fn = MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            dct_type=dct_type,
            norm=norm,
            log_mels=log_mels,
            melkwargs={
                "n_fft": n_fft,
                "win_length": win_length,
                "hop_length": hop_length,
                "f_min": f_min,
                "f_max": f_max,
                "n_mels": n_mels,
                "window_fn": functools.partial(get_stft_window, window),
            },
        )

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        x_mfcc, y_mfcc = self.mfcc_fn(x), self.mfcc_fn(y)
        return spectral_convergence_loss(x_mfcc, y_mfcc), log_stft_magnitude_loss(x_mfcc, y_mfcc)
