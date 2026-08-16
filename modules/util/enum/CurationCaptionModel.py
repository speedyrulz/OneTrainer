from enum import Enum


class CurationCaptionModel(Enum):
    """Which of OneTrainer's captioning models rewrites the captions of stuck images."""

    BLIP2 = 'BLIP2'
    BLIP = 'BLIP'
    WD14_VIT_2 = 'WD14_VIT_2'
    QWEN3_VL_4B = 'QWEN3_VL_4B'

    def __str__(self):
        return self.value
