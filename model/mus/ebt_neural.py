"""
Energy-Based Transformer for Music (Symbolic MIDI Tokens)

Adapted from NLP EBT but rewritten cleanly for music generation.
Uses MCMC-style iterative refinement to generate high-quality MIDI sequences.
"""

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L

import math
import os
from model.model_utils import *
from model.replay_buffer import CausalReplayBuffer


class EBT_MUS_NEUR(nn.Module):
    """
    Energy-Based Transformer for Music Generation (Neural)
    """
    #TODO: add logic for EBT_MUS_NEUR here (likely similar to video approach)
    pass
    