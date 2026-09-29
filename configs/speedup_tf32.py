"""The FlashSinkhorn TF32 units of configs/speedup.py only (alternating and
symmetric), timed like the full benchmark. Its units and keys are those of the
speedup config, so it can write into the same output directory
(SPEEDUP_OUTPUT_DIR) and the full run skips them on resume.

    SPEEDUP_OUTPUT_DIR=output/speedup_timed_rounded python run.py --config speedup_tf32 --count
"""

import dataclasses

from configs import speedup


def build_config():
    return dataclasses.replace(
        speedup.build_config(), flash_tf32=[True], no_srot=True, no_sinkslot=True,
        no_sinkslotcuda=True, no_sinkslotcuda_symmetric=True, no_sparsink=True, no_geomloss=True)


def __getattr__(name):
    if name == "CONFIG":
        return build_config()
    raise AttributeError(name)
