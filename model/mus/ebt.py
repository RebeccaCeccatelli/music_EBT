import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import torch.optim as optim
from torchmetrics import Accuracy

from transformers import AutoTokenizer

import math
import random
import os
from model.model_utils import *
from model.replay_buffer import CausalReplayBuffer


class EBT_MUS(L.LightningModule):
    # TODO: add logic for EBT_MUS here, this will likely be similar to EBT_NLP but with music specific changes
    pass