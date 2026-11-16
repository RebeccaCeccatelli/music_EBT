import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import torch.optim as optim
from torchmetrics import Accuracy
import traceback
from torchvision.transforms import functional as TF
import torchvision.models as models
import math
import random
import numpy as np
from functools import partial
from PIL import Image
import torchvision
from torchvision.transforms import ToPILImage
from datetime import datetime
import torch.distributed as dist
from typing import Optional, Tuple
from dataclasses import dataclass
import os
from torchvision.utils import make_grid, save_image
import json


@dataclass
class EBTModelArgs:
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    dyt_alpha_init: float = 0.5
    max_batch_size: int = 64
    max_seq_len: int = 16
    weight_initialization: str = "xavier"
    adaln_zero_init: bool = True
    ebt_norm: str = "rms"
    ebt_act_func: str = "silu"
    weight_initialization_gain: float = 1.0

model_sizes = { # small -> xl same as mamba https://arxiv.org/pdf/2312.00752; all others estimated empirically. LRs based off mamba where applicable
    "4xs": { # LR 0.0024 recommended
        "num_transformer_blocks": 2,
        "multiheaded_attention_heads": 2,
        "embedding_dim": 128,
    },
    "3xs": { # LR 0.0018
        "num_transformer_blocks": 4,
        "multiheaded_attention_heads": 4,
        "embedding_dim": 256,
    },
    "xxs": { # LR 0.0012
        "num_transformer_blocks": 6,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "2xs": { # LR 0.0012
        "num_transformer_blocks": 6,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "xs": { # LR 0.0009
        "num_transformer_blocks": 12,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "small": { # LR 0.0006
        "num_transformer_blocks": 12,
        "multiheaded_attention_heads": 12,
        "embedding_dim": 768,
    },
    "medium": { # 0.0003
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 16,
        "embedding_dim": 1024,
    },
    "large": { # 0.00025
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 16,
        "embedding_dim": 1536,
    },
    "xl": { # 0.0002
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 32,
        "embedding_dim": 2048,
    }
}




class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        out = self.linear(x)
        out = self.relu(out)
        out = self.dropout(out)
        return x + out  # Add the residual connection

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, final_size, dropout_rate, layer_norm, num_hidden_layers=1):
        super(MLP, self).__init__()
        self.add_residual_connections = True  # Residual connections are always on by default
        self.layers = nn.ModuleList()

        # Initial layer
        self.layers.append(nn.Linear(input_size, hidden_size, bias=False))
        if layer_norm:
            self.layers.append(nn.LayerNorm(hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(1, num_hidden_layers - 1):
            add_residual = self.add_residual_connections and i % 2 == 0

            if add_residual:
                self.layers.append(ResidualBlock(hidden_size, dropout_rate))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
                self.layers.append(nn.ReLU())

            self.layers.append(nn.Dropout(dropout_rate))

        # Last layer
        if final_size == hidden_size and self.add_residual_connections and (num_hidden_layers - 1) % 2 == 0:
            self.layers.append(ResidualBlock(hidden_size, dropout_rate))
        else:
            self.layers.append(nn.Linear(hidden_size, final_size, bias=False))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class Memory_Augmented_MLP(nn.Module):
    def __init__(self, input_modality_size, memory_size, input_modality_hidden_size, final_size, dropout_rate, layer_norm, num_hidden_layers=1):
        super(Memory_Augmented_MLP, self).__init__()
        self.add_residual_connections = True  # Residual connections are always on by default
        self.input_layers = nn.ModuleList()
        self.memory_size = memory_size

        # Initial layer
        self.input_layers.append(nn.Linear(input_modality_size, input_modality_hidden_size, bias=False))
        if layer_norm:
            self.input_layers.append(nn.LayerNorm(input_modality_hidden_size))
        self.input_layers.append(nn.ReLU())
        self.input_layers.append(nn.Dropout(dropout_rate))

        self.memory = nn.Embedding(1, memory_size)

        self.fusion_layers = nn.ModuleList()

        self.fusion_hidden_size = input_modality_hidden_size + memory_size

        # Hidden layers
        for i in range(1, num_hidden_layers - 1):
            add_residual = self.add_residual_connections and i % 2 == 0

            if add_residual:
                self.fusion_layers.append(ResidualBlock(self.fusion_hidden_size, dropout_rate))
            else:
                self.fusion_layers.append(nn.Linear(self.fusion_hidden_size, self.fusion_hidden_size, bias=False))
                self.fusion_layers.append(nn.ReLU())

            self.fusion_layers.append(nn.Dropout(dropout_rate))

        # Last layer
        if final_size == self.fusion_hidden_size and self.add_residual_connections and (num_hidden_layers - 1) % 2 == 0:
            self.fusion_layers.append(ResidualBlock(self.fusion_hidden_size, dropout_rate))
        else:
            self.fusion_layers.append(nn.Linear(self.fusion_hidden_size, final_size, bias=False))

    def forward(self, x):
        batch_size, sequence_length = x.shape[0], x.shape[1]
        for layer in self.input_layers:
            x = layer(x)
        
        memory_embedding = self.memory.weight[0]
        memory_embedding = memory_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, sequence_length, self.memory_size)
        fused_x = torch.cat((x, memory_embedding), dim=-1) # B, S, H + M (where H is hidden size M is memory size)
        
        for layer in self.fusion_layers:
            fused_x = layer(fused_x)
        
        return fused_x
    
class Memory_Gating_MLP(nn.Module):
    def __init__(self, input_modality_size, embedding_dim, process_memory_type = "add", process_memory_linear_layer = False):
        super(Memory_Gating_MLP, self).__init__()
        self.vocab_to_embed = nn.Linear(input_modality_size, embedding_dim, bias = False)
        self.memory = nn.Embedding(1, embedding_dim)
        self.process_memory_linear_layer = process_memory_linear_layer
        if self.process_memory_linear_layer:
            self.memory_linear_layer = nn.Linear(embedding_dim, embedding_dim, bias = False)
        self.process_memory_type = process_memory_type

    def forward(self, x):
        vocab_embeds = self.vocab_to_embed(x)
        memory_embedding = self.memory.weight[0]
        memory_embedding = memory_embedding.unsqueeze(0).unsqueeze(0).expand_as(vocab_embeds)
        if self.process_memory_linear_layer:
            memory_embedding = self.memory_linear_layer(memory_embedding)
        
        if self.process_memory_type == "add":
            final_embeddings = vocab_embeds + memory_embedding
        elif self.process_memory_type == "gate":
            final_embeddings = vocab_embeds * memory_embedding
        elif self.process_memory_type == "residual_gate":
            final_embeddings = vocab_embeds + vocab_embeds * memory_embedding
        else:
            raise NotImplementedError(f"self.process_memory_type {self.process_memory_type} not yet implemented")

        return final_embeddings


def calc_out_of_bounds_loss(energy): # gives loss for < 0 or > 1
    lower_bound_loss = torch.abs(energy)
    upper_bound_loss = torch.abs(energy - 1)
    loss = torch.where(energy < 0, lower_bound_loss, 
                    torch.where(energy > 1, upper_bound_loss, torch.zeros_like(energy)))
    loss = torch.mean(loss)
    
    return loss

def scale_clamp(tensor, min_value, max_value):
    scale_factor = torch.ones_like(tensor)
    scale_factor = torch.where(tensor > max_value, tensor / max_value, scale_factor)
    scale_factor = torch.where(tensor < min_value, tensor / min_value, scale_factor)
    
    scaled_tensor = tensor / scale_factor
    return scaled_tensor

def load_trained_pl_model(ckpt_path, new_hparams, for_inference = False):
    from base_model_trainer import ModelTrainer
    checkpoint = torch.load(ckpt_path, weights_only=False)
    model = ModelTrainer(new_hparams)
    model.load_state_dict(checkpoint['state_dict'])
    if for_inference:
        model.cuda().eval()
        model.model.eval()
    return model.model

def print_model_layers_and_status(model):
    for name, module in model.named_modules():
        print(f'Layer: {name}, Type: {type(module).__name__}, Training Mode: {module.training}')

def init_whole_model_weights(model, weight_initialization_method, nonlinearity='linear', weight_initialization_gain=1.0):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if weight_initialization_method == "he":
                valid_nonlinearities = ['linear', 'relu', 'leaky_relu', 'selu', 'tanh']
                if nonlinearity not in valid_nonlinearities:
                    raise ValueError(f"Unsupported nonlinearity: {nonlinearity}. Must be one of {valid_nonlinearities}")
                
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
                if weight_initialization_gain != 1.0:
                    m.weight.data *= weight_initialization_gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif weight_initialization_method == "xavier":
                nn.init.xavier_normal_(m.weight)
                if weight_initialization_gain != 1.0:
                    m.weight.data *= weight_initialization_gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            else:
                raise ValueError(f"Unknown weight init method: {weight_initialization_method}")
    
    model.apply(init_weights)


def hinged_mse_loss(predictions, targets, margin=0.1):
    """
    Compute the Hinged MSE loss between predictions and targets.
    :param predictions: Predicted values.
    :param targets: Ground truth values.
    :param margin: The threshold below which errors are ignored.
    :return: Hinged MSE loss.
    """
    errors = torch.abs(predictions - targets)
    hinged_errors = torch.where(errors > margin, errors, torch.zeros_like(errors))
    loss = torch.mean(hinged_errors ** 2)
    return loss

def setup_ebt(hparams): # specifically for EBT not for baseline transformer
    # to prevent circular import
    from model.ar_ebt_default import EBTDefault
    from model.ar_ebt_time_embed import EBTTimeConcat
    from model.ar_ebt_adaln import EBTAdaLN
    max_seq_len = hparams.context_length+1 # for next pred in context 
    max_seq_len = max_seq_len + 1 if hparams.ebt_type == "time_embed" else max_seq_len # need +1 since cat time embed on sequence dim

    adaln_zero_init = True if hparams.ebt_type == "adaln_zero" else False
    transformer_args = EBTModelArgs(dim = hparams.embedding_dim, n_layers = hparams.num_transformer_blocks, n_heads = hparams.multiheaded_attention_heads, max_batch_size = hparams.batch_size_per_device, max_seq_len=max_seq_len, weight_initialization = hparams.weight_initialization_method, adaln_zero_init=adaln_zero_init, ebt_norm=hparams.ebt_norm, ffn_dim_multiplier=hparams.ffn_dim_multiplier, ebt_act_func=hparams.ebt_act_func, weight_initialization_gain=hparams.weight_initialization_gain, dyt_alpha_init=hparams.dyt_alpha_init)
    
    if hparams.ebt_type == "default": # causal decoder trans for ebm https://arxiv.org/abs/2406.08862
        ebt = EBTDefault(params=transformer_args)
    elif hparams.ebt_type == "time_embed": # time embed
        ebt = EBTTimeConcat(params=transformer_args, max_mcmc_steps = hparams.mcmc_num_steps)
    else: # adaln or adaln_zero
        ebt = EBTAdaLN(params=transformer_args, max_mcmc_steps = hparams.mcmc_num_steps)

    return ebt

def setup_transformer(hparams): # specifically for baseline transformer
    from model.ar_transformer import Transformer, TransformerModelArgs
    transformer_args = TransformerModelArgs(dim = hparams.embedding_dim, n_layers = hparams.num_transformer_blocks, n_heads = hparams.multiheaded_attention_heads, max_batch_size = hparams.batch_size_per_device, max_seq_len=hparams.context_length, weight_initialization = hparams.weight_initialization_method, ffn_dim_multiplier=hparams.ffn_dim_multiplier, weight_initialization_gain=hparams.weight_initialization_gain)
    transformer = Transformer(params=transformer_args)
    return transformer

def has_layer_norm(model):
    return any(isinstance(module, nn.LayerNorm) for _, module in model.named_modules())

def init_wandb_watch(wandb_logger, model_trainer, wandb_watch_log_freq):
    if not has_layer_norm(model_trainer.model):
        wandb_logger.watch(model_trainer.model, log="all", log_freq = wandb_watch_log_freq)
    
    else: # all of complex below code is to get around the issue where wandb watch with layer norm has 'AttributeError: 'NoneType' object has no attribute 'data'' when logging gradients...
        non_layernorm_container = nn.Module()
        layernorm_container = nn.Module()

        non_ln_modules = {}
        ln_modules = {}

        for name, module in model_trainer.model.named_modules():
            if name == "": # skips top level model
                continue
            safe_name = name.replace(".", "_") # model cant contain '.' in name

            if isinstance(module, nn.LayerNorm):
                ln_modules[safe_name] = module
            else:
                # Only add modules that don't contain LayerNorm as submodules
                has_ln_child = any(isinstance(child, nn.LayerNorm) 
                                for child in module.modules())
                if not has_ln_child:
                    non_ln_modules[safe_name] = module

        for name, module in non_ln_modules.items():
            non_layernorm_container.add_module(name, module)

        for name, module in ln_modules.items():
            layernorm_container.add_module(name, module)

        # print("\nNon-LayerNorm modules:")
        # for name, _ in non_layernorm_container.named_modules():
        #     if name != "":  # Skip the container itself
        #         print(f"  - {name}")

        # print("\nLayerNorm modules:")
        # for name, _ in layernorm_container.named_modules():
        #     if name != "":  # Skip the container itself
        #         print(f"  - {name}")

        wandb_logger.watch(non_layernorm_container, log="all", log_freq=wandb_watch_log_freq)
        wandb_logger.watch(layernorm_container, log="parameters", log_freq=wandb_watch_log_freq)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
