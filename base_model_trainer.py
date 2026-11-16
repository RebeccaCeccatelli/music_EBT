import pytorch_lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torch.distributed import all_reduce
import wandb
import gc
import numpy as np

# Music-only trainer. The image/text/video code inherited from the original EBT
# codebase was removed after the thesis; see the `thesis-submission` tag for it.
from data.mus.symbolic.dataloaders.giga_midi_dataloader import GigaMIDIDataset
from data.mus.symbolic.dataloaders.custom_music_dataloader import CustomMusicDataset
from data.mus.symbolic.dataloaders.giga_midi_miditok_dataset import GigaMIDIMiditokDataset

from model.mus.ebt_symbolic import EBT_MUS_SYMB
from model.mus.ebt_neural import EBT_MUS_NEUR
from model.mus.baseline_llama_transformer import Baseline_Llama_Transformer_MUS
from model.mus.baseline_hf_gpt2_transformer import Baseline_HF_GPT2_Transformer_MUS

from inference.mus.generate_music import generate_music
from optimization import (WarmUpCosineAnnealingLR, LARS, exclude_bias_and_norm, StableAdamW, StableAdamWUnfused)
from utils import text_logger
from utils.metrics_calculator import get_torchmetrics


def _anticipation_dir_name(tokenizer_type: str) -> str:
    """Map tokenizer_type to the anticipation data subdirectory name.

    'anticipation/' holds AMT-style data: arrival-time, full vocab (55028 tokens),
    augment=10 (event + control tokens interleaved). This is what the AMT paper trains on.
    'anticipation-vanilla/' holds arrival-time data without the control block (smaller vocab, plain AR).
    """
    t = tokenizer_type.lower()
    if t in ('anticipation', 'anticipation-arrival-time'):
        return 'anticipation'
    return t  # 'anticipation-interarrival' etc.


class ModelTrainer(L.LightningModule):
    def __init__(self, hparams, trained_model = None):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))
        # self.txt_logger = hparams.txt_logger if txt_logger == None else txt_logger # txt_logger is no longer supported
        if self.hparams.modality not in ["MUS_SYMB", "MUS_NEUR"]:
            raise ValueError(f"Only music modalities (MUS_SYMB, MUS_NEUR) are supported, got {self.hparams.modality}. "
                             "Image/text/video support lives at the `thesis-submission` tag.")
        if self.hparams.modality in ["MUS_SYMB", "MUS_NEUR"]:
            if "execution_mode" in self.hparams and "save_generation_logs_dir" in self.hparams and self.hparams.execution_mode == "inference":
                print("setting up music inference logger")
                self.infer_logger = text_logger.setup_jsonl_logger(log_filename = "generations.jsonl", base_log_dir=self.hparams.save_generation_logs_dir)
                self.num_generated_music_samples = 0
        if self.hparams.modality in ["MUS_SYMB", "MUS_NEUR"]:
            # Music-specific setup handled in model initialization
            pass

        self.to_pil = ToPILImage()
        self.full_ds = None
        
        if trained_model is not None:
            self.model = trained_model
        else:
            if self.hparams.model_name == "ebt":
                if self.hparams.modality == "MUS_SYMB":
                    self.model = EBT_MUS_SYMB(self.hparams)
                elif self.hparams.modality == "MUS_NEUR":
                    self.model = EBT_MUS_NEUR(self.hparams)
                else:
                    raise ValueError(f"Modality: {self.hparams.modality} not supported as a base model trainer model as of now")
            elif self.hparams.model_name == "baseline_transformer":
                raise ValueError(f"For music, use --model_name baseline_llama_transformer or baseline_hf_gpt2_transformer instead of baseline_transformer")
            elif self.hparams.model_name == "baseline_llama_transformer":
                if self.hparams.modality in ["MUS_SYMB", "MUS_NEUR"]:
                    self.model = Baseline_Llama_Transformer_MUS(self.hparams)
                else:
                    raise ValueError(f"Baseline Llama transformer only supported for music modalities, got {self.hparams.modality}")
            elif self.hparams.model_name == "baseline_hf_gpt2_transformer":
                if self.hparams.modality in ["MUS_SYMB", "MUS_NEUR"]:
                    self.model = Baseline_HF_GPT2_Transformer_MUS(self.hparams)
                else:
                    raise ValueError(f"Baseline HF GPT2 transformer only supported for music modalities, got {self.hparams.modality}")
            else:
                raise ValueError(f"do not recognize model name: {self.hparams.model_name}")
        
        if self.hparams.compile_model:
            self.model = torch.compile(self.model, fullgraph=True)

        phases = ['train', 'valid', 'test']
        self.torchmetrics_dict = nn.ModuleDict()
        self.metrics = []
        for metric in self.hparams.metrics_list:
            self.metrics.append(metric)
        if len(self.metrics) > 0:
            assert self.hparams.num_classes != -1, "please set num_classes to the appropriate amount for the in use metrics. if are using accuracy and num_classes varies just set it to something that makes sense (shouldnt matter in that case)"
            assert self.hparams.metrics_task != "", "please set metrics_task to the appropriate value for your metrics"
        for phase in phases:
            for metric in self.metrics:
                self.torchmetrics_dict[f"{phase}_{metric}"] = get_torchmetrics(metric, self.hparams.metrics_average_type, self.hparams.num_classes, self.hparams.metrics_task)

        if self.hparams.wandb_watch:
            for name, module in self.model.named_modules(): # for activation logging
                module.name = name

        
    def on_train_start(self):
        if self.hparams.debug_unused_parameters:
            for name, param in self.model.named_parameters():
                if param.requires_grad and "image_encoder" not in name: # NOTE need to modify this code to exclude specific frozen portions
                    print(f"registering param - {name}")
                    param.register_hook(self.create_hook(name))
                    # TODO: consider adding exclusion logic for audio_encoder freezing in the future (if audio mus modality is added)
                else:
                    self.model.parameters_not_to_check.add(name)

        if self.global_step > 0 and hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            if hasattr(self.lr_scheduler, 'last_epoch'):
                self.lr_scheduler.last_epoch = self.global_step
                print(f"Resuming training: Set LR scheduler to step {self.global_step}")

    def on_save_checkpoint(self, checkpoint):
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state'] = self.lr_scheduler.state_dict()
            checkpoint['global_step_at_checkpoint'] = self.global_step

    def on_load_checkpoint(self, checkpoint):
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            if 'lr_scheduler_state' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
        # PL runs a full validation pass before training starts on resume. For EBT this
        # produces an unreliable (often way too low) loss because the fresh torch.randn
        # MCMC initial noise and fresh dataloader state are unrepresentative of normal
        # training — confirmed to also affect baseline models on the Anticipation
        # dataloader (every single restart produced a spurious valid_loss~0.2
        # checkpoint that never recurred). Flag it; validation_step() checks this and
        # neutralizes just the monitored value for that one cycle — see its comment
        # for why (and why skipping logging outright, tried first, is unsafe).
        self._skip_first_val_checkpoint = True

    def create_hook(self, name): #this is only used for debugging with `debug_unused_parameters`
        def hook(grad):
            self.model.used_parameters.add(name)  # Adjusted to self.model.used_parameters
        return hook
    
    @staticmethod
    def wandb_activation_hook(run, step):
        """ Weights & Biases histogram activation hook. """
        def hook(module, input, output):
            if isinstance(output, tuple):
                pass # when tried to do things had bug AttributeError: 'list' object has no attribute 'detach'
            else:
                run.experiment.log(
                    {f"activations/{module.name}": wandb.Histogram(output.detach().cpu().float())}, 
                    step=step
                )

        return hook
    
    def training_step(self, batch, batch_idx):
        if not self.hparams.no_wandb and self.hparams.wandb_watch and self.global_step % self.hparams.wandb_watch_log_freq == 0: # activation logging
            hook_handles = []
            hook_function = self.wandb_activation_hook(run=self.logger, step=self.global_step)
            for module in self.model.modules():
                if any(param.requires_grad for param in module.parameters(recurse=False)): # only do for unfrozen params that are training
                    handle = module.register_forward_hook(hook_function)
                    hook_handles.append(handle)
            
            eval_step_dict = self.eval_step(batch, "train")
            for handle in hook_handles:
                handle.remove()

        else:
            eval_step_dict = self.eval_step(batch, "train")
        
        self.log_metrics(eval_step_dict, "train")
        return eval_step_dict['loss']   
    
    def on_after_backward(self):
        if self.hparams.log_gradients:
            total_norm = 0.0
            num_parameters = 0
            num_grads_exceeding_clip_val = 0
            total_gradients = 0 # this is different from num_parameters since .parameters is for tensors of params but doesnt count each invididual parameter
            for param in self.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm  # Add the norm value to the total sum
                    num_parameters += 1
                    
                    total_gradients += torch.numel(param.grad)
                    num_grads_exceeding_clip_val += torch.sum(param.grad.abs() > self.hparams.gradient_clip_val)
                    
            assert num_parameters > 0, "no gradients after backwards detected please investigate"
            average_norm = (total_norm / num_parameters).detach()
            percentage_clipped = ((num_grads_exceeding_clip_val / total_gradients) * 100).detach()              
            
            things_to_log = {} 
            things_to_log['avg_gradient_norms'] = average_norm
            things_to_log['pct_gradient_clipped'] = percentage_clipped
            self.log_metrics(things_to_log, "train", log_torchmetrics = False)
        
    def on_train_batch_end(self, outputs, batch, batch_idx):
        #NOTE when using this may need to explicitly add code like 'if "image_encoder" not in name' for frozen params (with requires_grad == False)
        if self.hparams.debug_unused_parameters:
            all_parameters = {name for name, _ in self.model.named_parameters()}
            unused_parameters = all_parameters - self.model.used_parameters - self.model.parameters_not_to_check
            
            print(f"number of parameters total: {len(all_parameters)}")
            print(f"number of unused_parameters: {len(unused_parameters)}")
            print(f"Unused parameters: {unused_parameters}")
            print(f"Used parameters: {self.model.used_parameters}")
        
        if self.hparams.manual_gc_collect_every_n_steps != -1:
            if self.global_step % self.hparams.manual_gc_collect_every_n_steps == 0:
                print("calling GC manually")
                gc.collect()            
           
    def on_train_epoch_end(self):
        if self.hparams.optimizer != "adamw": # e.g. for lars need to manually update epoch
            optimizer = self.trainer.optimizers[0]
            optimizer.update_epoch(self.current_epoch)   

    def on_validation_epoch_end(self):
        # Free fragmented GPU cache after validation so the first training step
        # doesn't OOM on resume (PL runs a full val pass before training begins).
        torch.cuda.empty_cache()
        # Safe to clear now: validation_step() already neutralized the monitored
        # value for every step in the cycle that just finished. Clearing here
        # means the NEXT validation cycle logs normally.
        if getattr(self, '_skip_first_val_checkpoint', False):
            self._skip_first_val_checkpoint = False

    def validation_step(self, batch, batch_idx):
        eval_step_dict = self.eval_step(batch, "valid")
        # First validation cycle after a resume: the computed loss is unreliable
        # (see on_load_checkpoint). Two earlier attempts tried to catch and
        # override/skip the resulting bad metric — both confirmed broken:
        # (1) overriding trainer.callback_metrics after the fact in
        # on_validation_epoch_end was clobbered by PL's own
        # update_eval_epoch_metrics() recomputing it right after; (2) a
        # dedicated Callback trying the same override in on_validation_end
        # ran too late relative to ModelCheckpoint's own save logic; (3)
        # skipping self.log_metrics(...) entirely for this cycle (tried right
        # before this version) was reproduced via a standalone PL harness to
        # crash the run outright — ModelCheckpoint unconditionally requires
        # its monitored key to exist in every validation cycle's logged
        # metrics and raises MisconfigurationException otherwise.
        # So: log everything normally, but substitute the monitored value
        # with the worst-possible sentinel for its mode. The key still
        # exists (no crash), and it can never be chosen as a new "best"
        # checkpoint, which is the actual thing we need to prevent.
        if getattr(self, '_skip_first_val_checkpoint', False):
            monitor_string = self.hparams.checkpoint_monitor_string  # e.g. "valid_loss"
            local_key = monitor_string[len("valid_"):] if monitor_string.startswith("valid_") else monitor_string
            if local_key in eval_step_dict:
                sentinel = float('inf') if self.hparams.checkpoint_monitor_mode == "min" else float('-inf')
                eval_step_dict[local_key] = sentinel
        self.log_metrics(eval_step_dict, "valid")

    def test_step(self, batch, batch_idx):
        if self.hparams.execution_mode == "inference":  
            if self.hparams.modality in ["MUS_SYMB", "MUS_NEUR"]:
                # Unified music generation (handles EBT, Baseline Llama, and Baseline HF GPT2)
                outputs = generate_music(self.model, batch, self.hparams)
                
                # Log prompts and generated music tokens together for comparison.
                for i, tokens in enumerate(outputs['generation_tokens']):
                    sample_idx = self.num_generated_music_samples + i if hasattr(self, 'num_generated_music_samples') else i
                    log_entry = {
                        'sample_idx': sample_idx,
                        'prompt_tokens': outputs.get('prompt_tokens', [[]])[i],
                        'generated_tokens': tokens,
                    }
                    if 'generation_logprobs' in outputs:
                        log_entry['logprobs'] = outputs['generation_logprobs'][i]
                    if hasattr(self, 'infer_logger'):
                        self.infer_logger.log_data(log_entry)
                if hasattr(self, 'num_generated_music_samples'):
                    self.num_generated_music_samples += len(outputs['generation_tokens'])
                metric_outputs = {
                    key: value for key, value in outputs.items()
                    if isinstance(value, (int, float, torch.Tensor))
                }
                if metric_outputs:
                    self.log_metrics(metric_outputs, "test")
            else:
                raise NotImplementedError(f"Inference mode not supported for modality {self.hparams.modality} yet")
        else: # all other modes just get metrics
            eval_step_dict = self.eval_step(batch, "test")
            self.log_metrics(eval_step_dict, "test")
            
    def eval_step(self, batch, phase):
        things_to_log = self.model.forward_loss_wrapper(batch, phase) # things_to_log will be a dict of various things being logged. it NEEDS TO contain the 'loss' key as this is used to backprop

        if len(self.metrics) > 0:
            raise NotImplementedError("Need to implement torchmetrics stuff, i.e. looping through self.torchmetrics_dict.keys(), checking to make sure 'phase in key', and updating based off predicted and labels i.e. self.torchmetrics_dict[key].update(logits, labels), more info https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html (just be careful make sure to detach logits before using them and only update current phase). recommended to possibly return things_to_log and logits from forward_loss_wrapper to do this easily")

        return things_to_log
    
    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self): # this is a PL hook that returns optimizer and lr scheduler
        if self.hparams.modality in ["MUS_SYMB", "MUS_NEUR"]:
            return self.configure_optimizers_mus()
        else:
            raise NotImplementedError(f"Modality {self.hparams.modality} does not have configure optimizers supported yet")
        
    def get_optimizer(self, optimizer_parameters): # function for once gotten optimizer_parameters to get optimizer, i.e. adamw, lars, etc
        if self.hparams.optimizer == "lars":
            lars_exclude_bias_and_norm = None if not self.hparams.lars_exclude_bias_bn_wd else exclude_bias_and_norm
            optimizer = LARS(optimizer_parameters, lr=self.hparams.peak_learning_rate, weight_decay=self.hparams.weight_decay, momentum=self.hparams.beta1, eta=self.hparams.lars_trust_coeff, weight_decay_filter=lars_exclude_bias_and_norm, lars_adaptation_filter=lars_exclude_bias_and_norm)
        elif self.hparams.optimizer == "stableadamw":
            optimizer = StableAdamWUnfused(optimizer_parameters, betas=[self.hparams.beta1, self.hparams.beta2])
        else:
            optimizer = torch.optim.AdamW(optimizer_parameters, betas=[self.hparams.beta1, self.hparams.beta2])
        return optimizer
    
    def on_warm_up_finished(self):
        if hasattr(self.model, 'warm_up_finished'):
            self.model.warm_up_finished()
            print("Warm up finished, calling self.model.warm_up_finished()")
        else:
            print("Warm up finished, no self.model.warm_up_finished() exists so not doing anything")
    
    def get_lr_scheduler(self, optimizer):
        cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_scheduling_steps - self.hparams.warm_up_steps, eta_min=self.hparams.peak_learning_rate / self.hparams.min_lr_scale)
        lr_scheduler = WarmUpCosineAnnealingLR(optimizer, warm_up_steps = self.hparams.warm_up_steps, warm_up_base_lr_divider = self.hparams.warm_up_base_lr_divider, cosine_scheduler=cosine_annealing_scheduler, warm_up_finished_func=self.on_warm_up_finished)
        return lr_scheduler
    
    def get_optimizer_scheduler_dict(self, optimizer_parameters):
        optimizer = self.get_optimizer(optimizer_parameters)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        self.lr_scheduler = lr_scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
            
    def configure_optimizers_mus(self):
        if self.hparams.model_name == "ebt":
            alpha_param = self.model.alpha
            other_params = [param for name, param in self.model.named_parameters() if not any(keyword in name for keyword in ['alpha'])]
            assert len(other_params) > 1, "Could not gather model params correctly please investigate"

            optimizer_parameters = [
                {'params': alpha_param, 'weight_decay': 0.0, 'lr': self.hparams.mcmc_step_size_lr_multiplier*self.hparams.peak_learning_rate},  # No weight decay for alpha
                {'params': other_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}  # Weight decay for other parameters
            ]
            return self.get_optimizer_scheduler_dict(optimizer_parameters)
            
        elif self.hparams.model_name in ["baseline_llama_transformer", "baseline_hf_gpt2_transformer"]:
            all_params = [param for _, param in self.model.named_parameters()]
            optimizer_parameters = [
                {'params': all_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}  # Weight decay for other parameters
            ]
            return self.get_optimizer_scheduler_dict(optimizer_parameters)
        
        else:
            raise NotImplementedError(f"havent implemented configure optimizers for model {self.hparams.model_name}")

    def setup(self, stage=None):
        # NOTE when passing stage into datasets/dataloaders use string rep not the stage param from this func since is a PL enum
        # Assign train/val datasets for use in dataloaders 
        assert self.hparams.test_split_pct == 0, "Haven't implemented nonzero value for test_split_pct yet"

        if stage == "fit":
            if self.hparams.dataset_name == "giga_midi":
                tokenizer_type = getattr(self.hparams, 'tokenizer_type', 'REMI')
                if tokenizer_type.lower().startswith('anticipation'):
                    dir_name = _anticipation_dir_name(tokenizer_type)
                    pad_id = getattr(self.model, 'pad_token_id', 0)
                    self.train_ds = CustomMusicDataset(self.hparams, 'giga-midi', split='train', tokenizer_type=dir_name, pad_token_id=pad_id)
                    self.val_ds = CustomMusicDataset(self.hparams, 'giga-midi', split='validation', tokenizer_type=dir_name, pad_token_id=pad_id)
                else:
                    self.train_ds = GigaMIDIMiditokDataset(self.hparams, split="train")
                    self.val_ds = GigaMIDIMiditokDataset(self.hparams, split="validation")
            else:
                # Try to load as custom dataset
                try:
                    tokenizer_type = getattr(self.hparams, 'dataset_music_tokenizer_type', 'anticipation')
                    self.train_ds = CustomMusicDataset(self.hparams, self.hparams.dataset_name, split='train', tokenizer_type=tokenizer_type)
                    self.val_ds = CustomMusicDataset(self.hparams, self.hparams.dataset_name, split='validation', tokenizer_type=tokenizer_type)
                except Exception as e:
                    raise NotImplementedError(f"Haven't implemented dataset {self.hparams.dataset_name} and cannot load as custom dataset: {e}")
            print(f"{self.hparams.dataset_name} length of train_dataset: {len(self.train_ds)} and val_dataset: {len(self.val_ds)}")
            
        # Assign test dataset for use in dataloader(s)
        elif stage == "test":
            if self.hparams.dataset_name == "giga_midi":
                tokenizer_type = getattr(self.hparams, 'tokenizer_type', 'REMI')
                if tokenizer_type.lower().startswith('anticipation'):
                    dir_name = _anticipation_dir_name(tokenizer_type)
                    pad_id = getattr(self.model, 'pad_token_id', 0)
                    self.test_ds = CustomMusicDataset(self.hparams, 'giga-midi', split='test', tokenizer_type=dir_name, pad_token_id=pad_id)
                else:
                    self.test_ds = GigaMIDIMiditokDataset(self.hparams, split="validation")
            else:
                # Try to load as custom dataset
                try:
                    tokenizer_type = getattr(self.hparams, 'dataset_music_tokenizer_type', 'anticipation')
                    self.test_ds = CustomMusicDataset(self.hparams, self.hparams.dataset_name, split='test', tokenizer_type=tokenizer_type)
                except Exception as e:
                    raise NotImplementedError(f"haven't implemented test split for dataset {self.hparams.dataset_name} and cannot load as custom dataset: {e}")
            print(f"{self.hparams.dataset_name} length of test_ds: {len(self.test_ds)}")
        else:
            raise ValueError(f"Unknown stage: {stage}, please investigate")
    
    def get_collate_fn(self):
        return None  # music datasets return fixed-length tensors, no collator needed
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size_per_device, num_workers=self.hparams.num_workers, persistent_workers=True, collate_fn = self.get_collate_fn(), pin_memory = True, drop_last = False, shuffle = not self.hparams.no_shuffle, prefetch_factor=self.hparams.prefetch_factor)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size_per_device, num_workers=self.hparams.num_workers, persistent_workers=True, collate_fn = self.get_collate_fn(), pin_memory = True, drop_last = False, shuffle = False, prefetch_factor=self.hparams.prefetch_factor)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size_per_device, num_workers=self.hparams.num_workers, persistent_workers=True, collate_fn = self.get_collate_fn(), pin_memory = True, drop_last = False, shuffle = False, prefetch_factor=self.hparams.prefetch_factor)

    def log_metrics(self, metrics_dict, phase, log_torchmetrics = True):
        # first log torchmetrics if there are any
        if log_torchmetrics and len(self.metrics) > 0:
            phase_dict = {key : value for key, value in self.torchmetrics_dict.items() if phase in key}
            self.log_dict(phase_dict, on_step = False, on_epoch = True) # for these always do on_epoch

        # log all other metrics in metrics_dict
        scalar_metrics = {}
        keys = list(metrics_dict.keys()) # Iterate over a copy of the keys to avoid modification issues during iteration
        for key in keys:
            value = metrics_dict[key]
            if 'image' in key: # images
                image = self.to_pil(value)
                wandb_image = wandb.Image(image, mode="RGB")
                self.logger.experiment.log({f'{phase}_{key}': wandb_image})

            elif 'video' in key: # videos
                video_np = value.cpu().numpy()
                assert video_np.ndim != 5, "video should not include batch dimension, either fix that or add support"
                if video_np.shape[1] in [1, 3]:
                    pass  # Axes are already correct
                elif video_np.shape[-1] in [1, 3]:
                    # If video_np is (frames, height, width, channels), transpose axes
                    video_np = video_np.transpose(0, 3, 1, 2)
                else:
                    raise ValueError(f"Unexpected video shape: {video_np.shape}")
                if video_np.dtype != np.uint8:
                    video_np = (video_np * 255).astype(np.uint8)
                wandb_video = wandb.Video(video_np, fps=4, format="mp4")
                self.logger.experiment.log({f'{phase}_{key}': wandb_video})

            elif 'audio' in key and self.hparams.modality == "MUS_NEUR":  # audio for neural music modality only
                audio_np = value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                sample_rate = self.hparams.get('audio_sample_rate', 16000)
                wandb_audio = wandb.Audio(audio_np, sample_rate=sample_rate)
                self.logger.experiment.log({f'{phase}_{key}': wandb_audio})

            elif isinstance(value, torch.Tensor) and value.numel() > 1: # histogram
                self.logger.experiment.log({f"{phase}_{key}": wandb.Histogram(value.detach().cpu())})

            elif isinstance(value, torch.Tensor) and value.dim() == 0: # two types of scalar, tensor (here) and int/float (below)
                scalar_metrics[f"{phase}_{key}"] = value.detach()
            elif isinstance(value, (int, float)):
                scalar_metrics[f"{phase}_{key}"] = value
            else:
                raise ValueError(f"unsupported type/format in log_metrics, type:, {type(value)}, key: {key}")

        if scalar_metrics:
            self.log_dict(scalar_metrics, sync_dist=True, prog_bar=True)
        
        if self.hparams.mcmc_step_size_learnable:
            self.log("Alpha_MCMC_Step_Size", self.model.alpha.detach())
        if self.hparams.langevin_dynamics_noise_learnable:
            self.log("Langevin dynamics step size", self.model.langevin_dynamics_noise_std.detach())
        if len(self.trainer.optimizers) == 0:
            pass # is during testing lr doesnt matter
        else:
            current_lr = self.trainer.optimizers[0].param_groups[-1]['lr'] # relies on lr for most of model being the last param
            self.log("Global_LR", current_lr)
