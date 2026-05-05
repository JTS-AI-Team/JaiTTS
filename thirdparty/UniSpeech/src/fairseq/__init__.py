# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import os
import sys

from thirdparty.UniSpeech.src.fairseq.logging import meters, metrics

try:
    from .version import __version__  # noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

__all__ = ["pdb"]

# backwards compatibility to support `from fairseq.X import Y`
from thirdparty.UniSpeech.src.fairseq.distributed import utils as distributed_utils
from thirdparty.UniSpeech.src.fairseq.logging import progress_bar  # noqa

sys.modules["fairseq.distributed_utils"] = distributed_utils
sys.modules["fairseq.meters"] = meters
sys.modules["fairseq.metrics"] = metrics
sys.modules["fairseq.progress_bar"] = progress_bar

# initialize hydra
from thirdparty.UniSpeech.src.fairseq.dataclass.initialize import hydra_init
hydra_init()

import thirdparty.UniSpeech.src.fairseq.criterions  # noqa
import thirdparty.UniSpeech.src.fairseq.distributed  # noqa
import thirdparty.UniSpeech.src.fairseq.models  # noqa
import thirdparty.UniSpeech.src.fairseq.modules  # noqa
import thirdparty.UniSpeech.src.fairseq.optim  # noqa
import thirdparty.UniSpeech.src.fairseq.optim.lr_scheduler  # noqa
import thirdparty.UniSpeech.src.fairseq.pdb  # noqa
import thirdparty.UniSpeech.src.fairseq.tasks  # noqa
import thirdparty.UniSpeech.src.fairseq.token_generation_constraints  # noqa

