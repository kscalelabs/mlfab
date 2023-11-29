"""Tests the data munging components of the multi-logger.

This is necessary for ensuring that the various downstream loggers recieve
log values with a canonical format, to avoid having to duplicate the munging
in each of them.
"""


import pytest
import torch

import mlfab


class DummyLogger(mlfab.LoggerImpl):
    def __init__(self) -> None:
        super().__init__()

        self.images: dict[str, mlfab.LogImage] = {}
        self.audios: dict[str, mlfab.LogAudio] = {}
        self.videos: dict[str, mlfab.LogVideo] = {}

    def write(self, line: mlfab.LogLine) -> None:
        self.images.update({kk: vv for _, v in line.images.items() for kk, vv in v.items()})
        self.audios.update({kk: vv for _, v in line.audios.items() for kk, vv in v.items()})
        self.videos.update({kk: vv for _, v in line.videos.items() for kk, vv in v.items()})


@pytest.mark.parametrize(
    "multi_samples, in_shape, out_shape",
    [
        (False, (100, 100), (1, 100, 100)),
        (False, (2, 100, 100), None),
        (False, (3, 100, 100), (3, 100, 100)),
        (True, (100, 100), None),
        (True, (1, 100, 100), (1, 100, 100)),
        (True, (2, 100, 75), (1, 100, 150)),
        (True, (1, 3, 100, 100), (3, 100, 100)),
    ],
)
def test_log_image(
    multi_samples: bool,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...] | None,
) -> None:
    dummy_logger = DummyLogger()
    logger = mlfab.Logger()
    logger.add_logger(dummy_logger)
    test_img = torch.randn(*in_shape)
    if multi_samples:
        logger.log_images("test", test_img, keep_resolution=True)
    else:
        logger.log_image("test", test_img, keep_resolution=True)

    if out_shape is None:
        with pytest.raises(ValueError):
            logger.write(mlfab.State.init_state())
        return
    else:
        logger.write(mlfab.State.init_state())
        assert dummy_logger.images["test"].pixels.shape == out_shape


@pytest.mark.parametrize(
    "multi_samples, in_shape, out_shape",
    [
        (False, (2, 100), (2, 100)),
        (False, (100,), (1, 100)),
        (False, (2, 2, 100), None),
        (False, (3, 100), None),
        (True, (2, 100), (1, 200)),
        (True, (2, 2, 100), (2, 200)),
        (True, (2, 3, 100), None),
        (False, (2, 1024), (2, 1024)),
        (True, (2, 1, 1024), (1, 2048)),
    ],
)
def test_log_audio(
    multi_samples: bool,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...] | None,
) -> None:
    dummy_logger = DummyLogger()
    logger = mlfab.Logger()
    logger.add_logger(dummy_logger)
    test_wav = torch.randn(*in_shape)
    if multi_samples:
        logger.log_audios("test", test_wav, log_spec=False)
    else:
        logger.log_audio("test", test_wav, log_spec=False)

    if out_shape is None:
        with pytest.raises(ValueError):
            logger.write(mlfab.State.init_state())
        return
    else:
        logger.write(mlfab.State.init_state())
        assert dummy_logger.audios["test"].frames.shape == out_shape


@pytest.mark.parametrize(
    "multi_samples, in_shape, out_shape",
    [
        (False, (2, 4096), (513, 17)),
        (True, (2, 1, 4096), (513, 34)),
    ],
)
def test_log_spectrogram(
    multi_samples: bool,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...] | None,
) -> None:
    dummy_logger = DummyLogger()
    logger = mlfab.Logger()
    logger.add_logger(dummy_logger)
    test_wav = torch.randn(*in_shape)
    if multi_samples:
        logger.log_spectrograms("test", test_wav, keep_resolution=True)
    else:
        logger.log_spectrogram("test", test_wav, keep_resolution=True)

    if out_shape is None:
        with pytest.raises(ValueError):
            logger.write(mlfab.State.init_state())
        return
    else:
        logger.write(mlfab.State.init_state())
        assert dummy_logger.images["test"].pixels[0].shape == out_shape


@pytest.mark.parametrize(
    "multi_samples, in_shape, out_shape",
    [
        (False, (5, 100, 100), (5, 1, 100, 100)),
        (False, (2, 100, 100, 100), None),
        (False, (5, 3, 100, 100), (5, 3, 100, 100)),
        (True, (5, 100, 100), None),
        (True, (5, 1, 100, 20), (1, 1, 100, 100)),
        (True, (5, 3, 100, 20), (3, 1, 100, 100)),
        (True, (5, 4, 3, 100, 20), (4, 3, 100, 100)),
    ],
)
def test_log_video(
    multi_samples: bool,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...] | None,
) -> None:
    dummy_logger = DummyLogger()
    logger = mlfab.Logger()
    logger.add_logger(dummy_logger)
    test_video = torch.randn(*in_shape)
    if multi_samples:
        logger.log_videos("test", test_video)
    else:
        logger.log_video("test", test_video)

    if out_shape is None:
        with pytest.raises(ValueError):
            logger.write(mlfab.State.init_state())
        return
    else:
        logger.write(mlfab.State.init_state())
        assert dummy_logger.videos["test"].frames.shape == out_shape


@pytest.mark.parametrize(
    "multi_samples, in_shape, out_shape",
    [
        (False, (100, 100), (1, None, 100)),
        (False, (5, 100, 100), None),
        (True, (1, 100, 100), (1, None, 100)),
        (True, (5, 100, 20), (1, None, 100)),
    ],
)
def test_log_labeled_image(
    multi_samples: bool,
    in_shape: tuple[int, ...],
    out_shape: tuple[int | None, ...] | None,
) -> None:
    dummy_logger = DummyLogger()
    logger = mlfab.Logger()
    logger.add_logger(dummy_logger)
    test_img = torch.randn(*in_shape)
    if multi_samples:
        logger.log_labeled_images("test", (test_img, ["Test image"] * in_shape[0]), keep_resolution=True)
    else:
        logger.log_labeled_image("test", (test_img, "Test image"), keep_resolution=True)

    if out_shape is None:
        with pytest.raises(ValueError):
            logger.write(mlfab.State.init_state())
        return
    else:
        logger.write(mlfab.State.init_state())
        logged_image = dummy_logger.images["test"].pixels
        assert len(logged_image.shape) == len(out_shape)
        for i, j in zip(logged_image.shape, out_shape):
            if j is not None:
                assert i == j
