from enum import Enum


class MultiConfigStateStorage(Enum):
    # candidate training states are written to <workspace>/multi_config/state as torch files.
    # Slower, but the peak memory cost is one state instead of one per candidate, and a crashed
    # run can be inspected afterwards.
    DISK = 'DISK'

    # candidate training states are kept in system RAM. Much faster, but needs
    # (candidates + 1) * (trainable weights + optimizer state) of free RAM.
    RAM = 'RAM'

    def __str__(self):
        return self.value
