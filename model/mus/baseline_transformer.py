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


class Baseline_Transformer_MUS(L.LightningModule):
    #TODO: add logic for Baseline_Transformer_MUS here, this will likely be similar to Baseline_Transformer_NLP but with music specific changes
    pass
    