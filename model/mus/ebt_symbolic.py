"""
Energy-Based Transformer for Music (Symbolic MIDI Tokens)

Adapted from the NLP EBT (model/nlp/ebt.py) for symbolic music generation.
Uses MCMC-style iterative refinement over token distributions.
"""

import torch
from torch import nn
from torch.nn import functional as F
from argparse import Namespace

import math
from model.model_utils import *
from model.replay_buffer import CausalReplayBuffer
from data.mus.symbolic.tokenization.tokenizer_utils import load_tokenizer


class EBT_MUS_SYMB(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        # Normalise to a plain dict so .get() works everywhere in this class.
        # setup_ebt() needs attribute access, so we pass Namespace(**self.hparams) there.
        if isinstance(hparams, dict):
            self.hparams = dict(hparams)
        else:
            self.hparams = vars(hparams)

        tokenizer_type = self.hparams.get('tokenizer_type', 'REMI')
        tokenizer_config_path = self.hparams.get('tokenizer_config_path', None)
        dataset_name = self.hparams.get('dataset_name', 'giga-midi')
        use_vanilla = self.hparams.get('use_vanilla', False)

        self.tokenizer, self.vocab_size, self.pad_token_id = load_tokenizer(
            tokenizer_type=tokenizer_type,
            tokenizer_config_path=tokenizer_config_path,
            dataset_name=dataset_name,
            use_vanilla=use_vanilla,
        )
        print(f"Loaded {tokenizer_type} tokenizer: vocab_size={self.vocab_size}, pad_token_id={self.pad_token_id}")

        emb_dim = self.hparams['embedding_dim']
        weight_init = self.hparams['weight_initialization_method']
        weight_init_gain = self.hparams['weight_initialization_gain']

        self.embeddings = nn.Embedding(self.vocab_size, emb_dim)
        init_whole_model_weights(self.embeddings, weight_init, weight_initialization_gain=weight_init_gain)

        # Used when normalize_initial_condition=False; always created for checkpoint compatibility.
        self.vocab_to_embed = nn.Linear(self.vocab_size, emb_dim, bias=False)
        init_whole_model_weights(self.vocab_to_embed, weight_init, weight_initialization_gain=weight_init_gain)
        # Freeze when unused so DDP doesn't complain about parameters with no gradient.
        if self.hparams.get('normalize_initial_condition', False):
            self.vocab_to_embed.requires_grad_(False)

        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.alpha = nn.Parameter(
            torch.tensor(float(self.hparams['mcmc_step_size'])),
            requires_grad=self.hparams['mcmc_step_size_learnable'],
        )
        self.langevin_dynamics_noise_std = nn.Parameter(
            torch.tensor(float(self.hparams['langevin_dynamics_noise'])),
            requires_grad=False,  # enabled after warm-up if langevin_dynamics_noise_learnable
        )

        # setup_ebt uses attribute access, so wrap the dict in a Namespace.
        self.transformer = setup_ebt(Namespace(**self.hparams))

        self.mcmc_replay_buffer = (
            self.hparams.get('mcmc_replay_buffer', False)
            and self.hparams.get('execution_mode', 'pretrain') != 'inference'
        )
        if self.mcmc_replay_buffer:
            replay_buffer_samples = int(
                self.hparams['batch_size_per_device']
                * self.hparams['mcmc_replay_buffer_sample_bs_percent']
            )
            self.replay_buffer = CausalReplayBuffer(
                max_size=self.hparams['mcmc_replay_buffer_size'],
                sample_size=replay_buffer_samples,
            )
            self.replay_buffer_samples = replay_buffer_samples

        if self.hparams.get('debug_unused_parameters', False):
            self.used_parameters = set()
            self.parameters_not_to_check = set()

        self.finished_warming_up = False

    # =========================================================================
    # FORWARD: MCMC iterative refinement  (mirrors model/nlp/ebt.py EBT_NLP)
    # =========================================================================

    def forward(self, x, start_pos=0, learning=True, return_raw_logits=False,
                replay_buffer_logits=None, no_randomness=True):
        predicted_distributions = []
        predicted_energies = []

        real_embeddings_input = self.embeddings(x)   # (B, S, D)
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        predicted_tokens = self._corrupt_embeddings(real_embeddings_input)  # (B, S, V)
        if replay_buffer_logits is not None:
            predicted_tokens[batch_size - replay_buffer_logits.shape[0]:] = replay_buffer_logits

        alpha = torch.clamp(self.alpha, min=0.0001)
        if not no_randomness and self.hparams.get('randomize_mcmc_step_size_scale', 1) != 1:
            scale = self.hparams.get('randomize_mcmc_step_size_scale', 1)
            expanded_alpha = alpha.expand(batch_size, seq_length, 1)
            low = alpha / scale
            high = alpha * scale
            alpha = low + torch.rand_like(expanded_alpha) * (high - low)

        langevin_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=1e-6)

        # --- build the list of (landscape_index, step) pairs to iterate ---
        mcmc_num_steps = self.hparams.get('mcmc_num_steps', 2)
        rand_steps = self.hparams.get('randomize_mcmc_num_steps', 0)
        rand_steps_min = self.hparams.get('randomize_mcmc_num_steps_min', 0)
        rand_final_only = self.hparams.get('randomize_mcmc_num_steps_final_landscape', False)

        mcmc_steps = []
        for step in range(mcmc_num_steps):
            if rand_steps > 0:
                if rand_final_only:
                    if step == (mcmc_num_steps - 1):
                        if no_randomness:
                            mcmc_steps.extend([step] * (rand_steps + 1))
                        else:
                            min_s = 1 if rand_steps_min == 0 else rand_steps_min
                            reps = torch.randint(min_s, rand_steps + 2, (1,)).item()
                            mcmc_steps.extend([step] * reps)
                    else:
                        mcmc_steps.append(step)
                else:
                    if no_randomness:
                        if step == (mcmc_num_steps - 1):
                            mcmc_steps.extend([step] * (rand_steps + 1))
                        else:
                            mcmc_steps.append(step)
                    else:
                        min_s = 1 if rand_steps_min == 0 else rand_steps_min
                        reps = torch.randint(min_s, rand_steps + 2, (1,)).item()
                        mcmc_steps.extend([step] * reps)
            else:
                mcmc_steps.append(step)

        normalize = self.hparams.get('normalize_initial_condition', False)
        normalize_only_first = self.hparams.get('normalize_initial_condition_only_first_step', False)
        truncate_mcmc = self.hparams.get('truncate_mcmc', False)
        no_mcmc_detach = self.hparams.get('no_mcmc_detach', False)
        langevin_noise = self.hparams.get('langevin_dynamics_noise', 0)
        clamp_grad = self.hparams.get('clamp_futures_grad', False)
        absolute_clamp = self.hparams.get('absolute_clamp', 0.0)
        sharpen = self.hparams.get('sharpen_predicted_distribution', 0.0)

        with torch.set_grad_enabled(True):
            for i, mcmc_step in enumerate(mcmc_steps):

                if no_mcmc_detach:
                    predicted_tokens.requires_grad_()
                else:
                    predicted_tokens = (
                        predicted_tokens.detach()
                        .requires_grad_()
                        .reshape(batch_size, seq_length, self.vocab_size)
                    )

                if langevin_noise != 0:
                    ld_noise = torch.randn_like(predicted_tokens.detach()) * langevin_noise_std
                    predicted_tokens = predicted_tokens + ld_noise

                # Convert logits → embeddings (matches NLP EBT logic, always matmul when normalized)
                if normalize:
                    if normalize_only_first:
                        if mcmc_step == 0:
                            predicted_tokens = self.softmax(predicted_tokens)
                    else:
                        predicted_tokens = self.softmax(predicted_tokens)
                    predicted_embeddings = torch.matmul(predicted_tokens, self.embeddings.weight)
                else:
                    predicted_embeddings = self.vocab_to_embed(predicted_tokens)

                all_embeddings = torch.cat((real_embeddings_input, predicted_embeddings), dim=1)  # (B, 2S, D)

                energy_preds = self.transformer(all_embeddings, start_pos=start_pos, mcmc_step=mcmc_step)
                energy_preds = energy_preds.reshape(-1, 1)
                predicted_energies.append(energy_preds)

                # truncate_mcmc: only build the full compute graph on the last step
                if truncate_mcmc:
                    create_graph = learning if i == (len(mcmc_steps) - 1) else False
                else:
                    create_graph = learning

                predicted_tokens_grad = torch.autograd.grad(
                    [energy_preds.sum()], [predicted_tokens], create_graph=create_graph
                )[0]

                if clamp_grad:
                    min_and_max = self.hparams.get('clamp_futures_grad_max_change', 9.0) / alpha
                    predicted_tokens_grad = torch.clamp(predicted_tokens_grad, min=-min_and_max, max=min_and_max)

                if torch.isnan(predicted_tokens_grad).any() or torch.isinf(predicted_tokens_grad).any():
                    raise ValueError("NaN or Inf gradients detected during MCMC.")

                predicted_tokens = predicted_tokens - alpha * predicted_tokens_grad

                if absolute_clamp != 0.0:
                    predicted_tokens = torch.clamp(predicted_tokens, min=-absolute_clamp, max=absolute_clamp)
                if sharpen != 0.0:
                    predicted_tokens = predicted_tokens / sharpen

                if return_raw_logits:
                    predicted_distributions.append(predicted_tokens)
                else:
                    predicted_distributions.append(
                        self.log_softmax(predicted_tokens).reshape(-1, self.vocab_size)
                    )

        return predicted_distributions, predicted_energies

    # =========================================================================
    # LOSS
    # =========================================================================

    def forward_loss_wrapper(self, batch, phase="train"):
        no_randomness = phase != "train"

        if not no_randomness and self.mcmc_replay_buffer:
            all_tokens = batch['input_ids'].squeeze(dim=1)
            input_ids, replay_buffer_logits, next_token_indices = self.replay_buffer.get_batch(all_tokens)
            predicted_distributions, predicted_energies = self(
                input_ids, return_raw_logits=True,
                replay_buffer_logits=replay_buffer_logits, no_randomness=False,
            )
            self.replay_buffer.update(all_tokens.detach(), predicted_distributions[-1].detach())
        else:
            input_ids = batch['input_ids'].squeeze(dim=1)[:, :-1]
            predicted_distributions, predicted_energies = self(
                input_ids, return_raw_logits=True, no_randomness=no_randomness,
            )
            next_token_indices = batch['input_ids'].squeeze(dim=1)[:, 1:]

        next_token_indices = next_token_indices.reshape(-1)  # (B*S,)

        reconstruction_loss = 0
        total_mcmc_steps = len(predicted_energies)
        initial_loss = initial_energy = final_loss = final_energy = None

        soften = self.hparams.get('soften_target_prob_dist', 0.0)
        truncate_mcmc = self.hparams.get('truncate_mcmc', False)

        for mcmc_step, (predicted_dist, energy) in enumerate(zip(predicted_distributions, predicted_energies)):
            # predicted_dist is (B, S, V) raw logits (return_raw_logits=True)
            if predicted_dist.dim() > 2:
                predicted_dist = predicted_dist.reshape(-1, self.vocab_size)

            if soften != 0.0:
                if total_mcmc_steps <= 1:
                    label_smoothing = 0.0
                else:
                    label_smoothing = ((total_mcmc_steps - 1) - mcmc_step) / (total_mcmc_steps - 1) * soften
                ce_loss = F.cross_entropy(
                    predicted_dist, next_token_indices,
                    ignore_index=self.pad_token_id, label_smoothing=label_smoothing,
                )
            else:
                ce_loss = F.cross_entropy(
                    predicted_dist, next_token_indices, ignore_index=self.pad_token_id,
                )

            if truncate_mcmc:
                if mcmc_step == (total_mcmc_steps - 1):
                    reconstruction_loss = ce_loss
                    final_loss = ce_loss.detach()
                    final_energy = energy.squeeze().mean().detach()
            else:
                reconstruction_loss = reconstruction_loss + ce_loss
                if mcmc_step == (total_mcmc_steps - 1):
                    final_loss = ce_loss.detach()
                    final_energy = energy.squeeze().mean().detach()

            if mcmc_step == 0:
                initial_loss = ce_loss.detach()
                initial_energy = energy.squeeze().mean().detach()

        if not truncate_mcmc:
            reconstruction_loss = reconstruction_loss / total_mcmc_steps

        energy_improvement = (
            (initial_energy - final_energy)
            if initial_energy is not None and final_energy is not None
            else torch.tensor(0.0)
        )
        perplexity = torch.exp(final_loss) if final_loss is not None else torch.tensor(0.0)

        return {
            'loss': reconstruction_loss,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'energy_improvement': energy_improvement,
            'perplexity': perplexity,
        }

    # =========================================================================
    # ADVANCED INFERENCE  (mirrors EBT_NLP.ebt_advanced_inference)
    # =========================================================================

    def ebt_advanced_inference(self, original_real_input_ids, start_pos=0, learning=False):
        real_embeddings_input = self.embeddings(original_real_input_ids)  # (B, S, D)
        original_predicted_tokens = self._corrupt_embeddings(real_embeddings_input)  # (B, S, V)

        infer_alpha = self.hparams.get('infer_ebt_override_alpha', 0)
        if infer_alpha >= 1:
            alpha = torch.tensor(float(infer_alpha), device=self.alpha.device)
        elif 0 < infer_alpha < 1:
            alpha = self.alpha * infer_alpha
        else:
            alpha = self.alpha

        infer_noise = self.hparams.get('infer_langevin_dynamics_noise', 0)
        if infer_noise != 0:
            noise = torch.tensor(
                float(infer_noise),
                dtype=self.langevin_dynamics_noise_std.dtype,
                device=self.langevin_dynamics_noise_std.device,
            )
        else:
            noise = self.langevin_dynamics_noise_std

        B, S, V = original_predicted_tokens.shape
        G = self.hparams.get('infer_generated_samples', 1)

        if G > 1:
            repeated_pred = self._corrupt_embeddings(real_embeddings_input.repeat_interleave(G, dim=0))
            repeated_real = real_embeddings_input.repeat_interleave(G, dim=0)
            repeated_bs = B * G
        else:
            repeated_pred = original_predicted_tokens
            repeated_real = real_embeddings_input
            repeated_bs = B

        all_final_pred = torch.zeros_like(repeated_pred)
        energies_accum = None
        dists_accum = None

        chunk_size = B
        for start in range(0, repeated_bs, chunk_size):
            end = min(start + chunk_size, repeated_bs)
            final_chunk, energies_chunk, dists_chunk = self._run_ebt_inference_steps(
                repeated_pred[start:end], repeated_real[start:end], alpha, noise, start_pos, learning,
            )
            all_final_pred[start:end] = final_chunk
            energies_chunk = [e.reshape(end - start, -1) for e in energies_chunk]
            if energies_accum is None:
                energies_accum = list(energies_chunk)
                dists_accum = [p.detach() for p in dists_chunk]
            else:
                for k in range(len(energies_accum)):
                    energies_accum[k] = torch.cat([energies_accum[k], energies_chunk[k]], dim=0)
                for k in range(len(dists_accum)):
                    if k < len(dists_chunk):
                        dists_accum[k] = torch.cat([dists_accum[k], dists_chunk[k].detach()], dim=0)

        if G > 1:
            final_energies_3d = energies_accum[-1].reshape(B, G, S)
            technique = self.hparams.get('infer_energy_sampling_technique', 'min')
            if technique == 'min':
                best_idx = final_energies_3d.argmin(dim=1)
            elif technique == 'max':
                best_idx = final_energies_3d.argmax(dim=1)
            elif technique == 'max_gap':
                gap = energies_accum[0].reshape(B, G, S) - final_energies_3d
                best_idx = gap.argmax(dim=1)
            else:
                raise ValueError(f"Unknown infer_energy_sampling_technique: {technique}")

            all_final_pred_4d = all_final_pred.reshape(B, G, S, V)
            b_idx = torch.arange(B, device=all_final_pred.device).unsqueeze(-1)
            s_idx = torch.arange(S, device=all_final_pred.device).unsqueeze(0)
            final_output = all_final_pred_4d[b_idx, best_idx, s_idx, :]
        else:
            final_output = all_final_pred

        return final_output, energies_accum, dists_accum

    def _run_ebt_inference_steps(self, initial_pred_tokens, real_embeds, alpha, noise, start_pos, learning):
        energies_list = []
        pred_states_list = [initial_pred_tokens]

        normalize = self.hparams.get('normalize_initial_condition', False)
        normalize_only_first = self.hparams.get('normalize_initial_condition_only_first_step', False)
        clamp_grad = self.hparams.get('clamp_futures_grad', False)
        langevin_first_only = self.hparams.get('infer_langevin_first_step', False)

        def _to_embeds(tokens, step_idx):
            if normalize:
                if normalize_only_first:
                    t = self.softmax(tokens) if step_idx == 0 else tokens
                else:
                    t = self.softmax(tokens)
                return torch.matmul(t, self.embeddings.weight)
            else:
                return self.vocab_to_embed(tokens)

        def do_mcmc_step(step_idx, cur_pred, a):
            with torch.set_grad_enabled(True):
                cur_pred = cur_pred.detach().requires_grad_()
                if not langevin_first_only or step_idx == 0:
                    cur_pred = cur_pred + noise * torch.randn_like(cur_pred)
                pred_embeds = _to_embeds(cur_pred, step_idx)
                combined = torch.cat([real_embeds, pred_embeds], dim=1)
                energies = self.transformer(combined, start_pos=start_pos, mcmc_step=step_idx)
                energies = energies.reshape(-1)
                energies_list.append(energies.detach())
                grad = torch.autograd.grad(energies.sum(), [cur_pred], create_graph=learning)[0]
                if clamp_grad:
                    mv = self.hparams.get('clamp_futures_grad_max_change', 9.0) / a
                    grad = torch.clamp(grad, -mv, mv)
                return (cur_pred - a * grad).detach()

        def get_energy(step_idx, cur_pred):
            with torch.no_grad():
                pred_embeds = _to_embeds(cur_pred.detach(), step_idx)
                combined = torch.cat([real_embeds, pred_embeds], dim=1)
                energies = self.transformer(combined, start_pos=start_pos, mcmc_step=step_idx)
                return energies.reshape(-1)

        mcmc_num_steps = self.hparams.get('mcmc_num_steps', 2)
        ebt_type = self.hparams.get('ebt_type', 'default')
        infer_ebt_num_steps = self.hparams.get('infer_ebt_num_steps', 1)
        infer_steps_final = self.hparams.get('infer_steps_final_landscape', False)
        infer_alpha_final = self.hparams.get('infer_alpha_final_landscape', False)
        rand_steps_min = self.hparams.get('randomize_mcmc_num_steps_min', 0)

        pred_state = initial_pred_tokens
        if ebt_type == 'default':
            total_steps = infer_ebt_num_steps if infer_ebt_num_steps > 1 else mcmc_num_steps
            for step_idx in range(total_steps):
                pred_state = do_mcmc_step(step_idx, pred_state, alpha)
                pred_states_list.append(pred_state)
        else:
            for step_idx in range(mcmc_num_steps):
                is_final_landscape = (step_idx == mcmc_num_steps - 1)
                if infer_steps_final and not is_final_landscape:
                    a = self.alpha if infer_alpha_final else alpha
                    pred_state = do_mcmc_step(step_idx, pred_state, a)
                    pred_states_list.append(pred_state)
                else:
                    inner = infer_ebt_num_steps if infer_ebt_num_steps != 1 else max(rand_steps_min, 1)
                    for _ in range(inner):
                        a = self.alpha if (infer_alpha_final and not is_final_landscape) else alpha
                        pred_state = do_mcmc_step(step_idx, pred_state, a)
                        pred_states_list.append(pred_state)

        energies_list.append(get_energy(mcmc_num_steps - 1, pred_state))
        return pred_state, energies_list, pred_states_list

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _corrupt_embeddings(self, embeddings):
        batch_size, seq_length = embeddings.shape[0], embeddings.shape[1]
        condition = self.hparams.get('denoising_initial_condition', 'random_noise')
        if condition == 'random_noise':
            scale = self.hparams.get('gaussian_random_noise_scaling', 1)
            return (
                torch.randn(batch_size, seq_length, self.vocab_size, device=embeddings.device)
                * scale
            )
        elif condition == 'zeros':
            return torch.zeros(batch_size, seq_length, self.vocab_size, device=embeddings.device)
        else:
            raise NotImplementedError(f"denoising_initial_condition={condition} not supported")

    def warm_up_finished(self):
        if self.hparams.get('langevin_dynamics_noise_learnable', False):
            self.langevin_dynamics_noise_std.requires_grad = True
        if self.hparams.get('clamp_max_after_warm_up', 0.0) != 0.0:
            self.hparams['clamp_futures_grad_max_change'] = self.hparams['clamp_max_after_warm_up']
        self.finished_warming_up = True
