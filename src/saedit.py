import torch
from typing import List, Optional,Tuple
import torch.nn as nn
from diffusers import DiffusionPipeline
import numpy as np

CALLBACK_ON_STEP_END_INPUT_TENSORS = ['prompt_embeds']


class SAEditCallback:
    def __init__(
                 self, 
                 pipeline: DiffusionPipeline,
                 sae: nn.Module,
                 source_tokens_to_edit: List[str],
                 factor: float, 
                 sentence_pairs: List[Tuple[str, str]],
                 prompt: str = None,
                 max_sequence_length: int = 256, 
                 ratio_percentile: float = 99.5, 
                 min_target_percentile: float = 90.0,
                 aggregate: str = "pca"):
        """
        Args:
            pipeline: Inference pipeline/model with text encoder and tokenizer.
            sae (nn.Module): The sparse autoencoder model to use for embedding manipulation.
            source_tokens_to_edit (List[str]): List of tokens to edit in the source prompt.
            factor (float): An edit strength factor (typically between 0 and 1).
            sentence_pairs (List[Tuple[str, str]]): List of (source_sentence, target_sentence) pairs for direction finding.
            prompt (str, optional): The input prompt string to be edited.
            max_sequence_length (int, optional): Maximum sequence length for tokenization (default: 256).
            ratio_percentile (float, optional): Percentile for ratio thresholding (default: 99.5).
            min_target_percentile (float, optional): Percentile for minimum target activation threshold (default: 90.0).
            aggregate (str, optional): Aggregation method across sentence pairs ('pca' or 'mean', default: 'pca').
        """
        self.sae = sae.to(torch.bfloat16)
        self.cache = {}
        self.source_tokens_to_edit = source_tokens_to_edit
        self.prompt = prompt
        self.original_prompt_embeds = None
        self.pipeline = pipeline
        self.interpolation_factors = self.calculate_factor(factor)
        self.max_sequence_length = max_sequence_length
        self.global_dir_sae = self.set_global_direction_from_pairs_activation_mask(sentence_pairs=sentence_pairs,
                                                                                   ratio_percentile=ratio_percentile,
                                                                                   min_target_percentile=min_target_percentile,
                                                                                   aggregate=aggregate)

    @staticmethod
    def calculate_factor(factor):
        """
        Calculate interpolation factors for gradual editing.
        
        Args:
            factor: Base edit strength factor (typically 0-1 though can exceed 1 at some cases )
        
        Returns:
            Array of interpolation factors for each diffusion step
        
        Example:
            >>> factors = calculate_factor(0.5)
            >>> print(factors.shape)
            (40,)
        """
        factor *= 15
        limit = factor
        interpolation_factors = np.linspace(0, 40, num=40) / 40.
        interpolation_factors = np.exp(interpolation_factors * factor) - 1
        interpolation_factors[interpolation_factors > limit] = limit
        return interpolation_factors

    # ---------- token + SAE helpers ----------
    #
    def get_tokens(self, sentence):
        """
        Given a T5 tokenizer and a sentence, returns a list of tuples
        containing each token and its index in the tokenized sequence.

        Args:
            tokenizer: T5 tokenizer instance
            sentence (str): Input sentence to tokenize

        Returns:
            List[Tuple[str, int]]: List of (token, index) tuples
        """
        # Tokenize the sentence
        tokens = self.pipeline.tokenizer_2.tokenize(sentence)
        tokens = [token.strip("▁") for token in tokens]
        return tokens

    def get_sentence_sae_embed(self, embeds, sentence):
        if sentence in self.cache:
            return self.cache[sentence].clone()
        with torch.no_grad():
            sae_rep = self.sae.encode(embeds.cuda())
        self.cache[sentence] = sae_rep
        return sae_rep.clone()

    def get_idx_to_edit(self, sentence):
        tokens = self.get_tokens(sentence)
        idx_to_edit_list = []
        for i, token in enumerate(tokens):
            if token in self.source_tokens_to_edit:
                idx_to_edit_list.append(i)
        return idx_to_edit_list, tokens

    @torch.no_grad()
    def set_global_direction_from_pairs_activation_mask(
            self,
            sentence_pairs,  # List[Tuple[str, str]]
            ratio_percentile: float,  # keep top p% by ratio (ignored if topk used)
            min_target_percentile: float,  # also require t_pool to be “high”
            aggregate: str,  # "pca" or "mean"
            eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Learn a unit global direction using 'activation-only' masking:
          - Pool over tokens (max)
          - Ratio / log-ratio target vs. source
          - Select target-only neurons, keep either binary or t_pool weights
          - Aggregate across pairs via PCA (SVD) or mean
        """
        sae_dev = next(self.sae.parameters()).device
        selected = []

        for src, tgt in sentence_pairs:
            # 1) get T5 embeddings exactly as generation uses
            h_src = self.pipeline._get_t5_prompt_embeds(prompt=[src], max_sequence_length=self.max_sequence_length)
            h_tgt = self.pipeline._get_t5_prompt_embeds(prompt=[tgt], max_sequence_length=self.max_sequence_length)

            # 2) SAE encode -> [1, T, D] -> squeeze token dim
            a_src = self.sae.encode(h_src.to(sae_dev)).squeeze(0)
            a_tgt = self.sae.encode(h_tgt.to(sae_dev)).squeeze(0)

            # Tokenize to get attention masks
            tok_src = self.pipeline.tokenizer_2(
                [src],
                max_length=self.max_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            )

            tok_tgt = self.pipeline.tokenizer_2(
                [tgt],
                max_length=self.max_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            )

            m_src = tok_src["attention_mask"][0].to(a_src.device).bool()  # [S]
            m_tgt = tok_tgt["attention_mask"][0].to(a_tgt.device).bool()  # [S]

            # Trim at the first PAD (assumes mask is 111...000)
            L_src = int(m_src.long().sum().item())
            L_tgt = int(m_tgt.long().sum().item())


            src_tokens = self.get_tokens(src)[:L_src]
            tgt_tokens = self.get_tokens(tgt)[:L_tgt]

            # Final indices
            src_idxs = list(range(L_src))
            tgt_idxs = list(range(L_tgt))

            # 3) max-pool chosen tokens -> [D] # TODO: maybe we should use mean?
            # do math in float32 for numerical stability
            s_pool = a_src[src_idxs, :].max(dim=0).values.float()
            t_pool = a_tgt[tgt_idxs, :].max(dim=0).values.float()

            ratio = t_pool / (s_pool + eps)
            ratio /= ratio.max()

            # 5) build mask of "target-only" neurons
            r_thr = torch.quantile(ratio, ratio_percentile / 100.0)
            # t_thr = torch.quantile(t_pool, min_target_percentile / 100.0)
            mask = (ratio >= r_thr) #& (t_pool >= t_thr)

            if not mask.any():
                continue

            # 6) build the masked vector for this pair
            # keep target activations on selected neurons if you want a tiny bit of weighting
            v = torch.zeros_like(t_pool)
            v[mask] = t_pool[mask]
            selected.append(v.unsqueeze(0))
            
        if not selected:
            raise ValueError("No pairs produced a usable activation mask. Relax thresholds or set topk.")

        M = torch.cat(selected, dim=0)  # [N_pairs, D]

        # 7) aggregate across pairs
        if aggregate.lower() == "pca":
            # X = M.float() - M.float().mean(dim=0, keepdim=True)
            X = M.float()
            # SVD on f32 for stability (works on CUDA)
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            g = Vh[0]  # [D]
            # align sign with average masked vector
            if torch.dot(g, M.float().mean(dim=0)) < 0:
                g = -g
        elif aggregate.lower() == "mean":
            g = M.mean(dim=0).float()
        else:
            raise ValueError(f"aggregate must be 'pca' or 'mean', got {aggregate!r}")
        return g.to(next(self.sae.parameters()).dtype).detach()

    # ---------- interpolate: use global dir ----------

    @torch.inference_mode()
    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        if self.global_dir_sae is None:
            raise RuntimeError(
                "Global direction not set. Call set_global_direction_from_pairs(...) first "
                "or pass global_dir_sae in __init__."
            )
        if step_index == 0:
            self.original_prompt_embeds = callback_kwargs['prompt_embeds']
        prompt_embeds = self.original_prompt_embeds
        a_src = self.get_sentence_sae_embed(prompt_embeds, self.prompt)
        idxs, _ = self.get_idx_to_edit(self.prompt)
        if idxs:
            g = self.global_dir_sae.to(a_src.device, dtype=a_src.dtype)
            step = self.interpolation_factors[step_index]
            for i in idxs:
                a_src[0, i, :] = a_src[0, i, :] + step * g
        new_emeds = self.sae.decode(a_src).to(prompt_embeds.dtype)
        callback_kwargs['prompt_embeds'] = new_emeds
        return callback_kwargs
        