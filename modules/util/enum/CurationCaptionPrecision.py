from enum import Enum


class CurationCaptionPrecision(Enum):
    """How the Qwen3-VL captioner is loaded, which is mostly a question of VRAM.

    The captioner loads while the training model is still resident, so its footprint comes on top of
    the run's. NF4 is the default: ~3.5GB instead of ~9GB, and captioning is robust to it.
    """

    NF4 = 'NF4'
    INT8 = 'INT8'
    BF16 = 'BF16'

    def __str__(self):
        return self.value
