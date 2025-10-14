import torch
from typing import List, Optional
import torch.nn as nn

callback_on_step_end_tensor_inputs = ["prompt_embeds"]


class SAEditCallback:
    def __init__(self, interpolation_factors: List[float], t: int, source_tokens_to_edit: List[str], sae: nn.Module,
                 t5_tokenizer, global_dir_sae: torch.Tensor = None):
        self.interpolation_factors = interpolation_factors
        self.t=t
        self.sae = sae.to(torch.bfloat16)
        self.t5_tokenizer = t5_tokenizer
        self.cache = {}
        self.source_tokens_to_edit = source_tokens_to_edit
        self.global_dir_sae = None
        if global_dir_sae is not None:
            self.global_dir_sae = (global_dir_sae / (global_dir_sae.norm(p=2) + 1e-8)).detach() # TODO: is norm needed?

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
        tokens = self.t5_tokenizer.tokenize(sentence)
        tokens = [token.strip("▁") for token in tokens]

        # Create list of tuples with token and its index
        #         token_index_pairs = [(token, idx) for idx, token in enumerate(tokens)]

        return tokens

    def get_sentence_sae_embed(self, embeds, sentence):
        # if sentence in self.cache:
        #     return self.cache[sentence].clone()
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
    def get_processed_embed(self, embed: torch.Tensor) -> torch.Tensor:
        sae_embed = self.get_sentence_sae_embed(embed, "bbp").to(embed.device)
        return self.sae.decode(sae_embed).to(embed.dtype)

    @torch.no_grad()
    def set_global_direction_from_pairs_activation_mask(
            self,
            pipeline,  # SAEFluxPipeline
            sentence_pairs,  # List[Tuple[str, str]]
            max_sequence_length: int = 226,
            *,
            use_log_ratio: bool = False,  # log((t+eps)/(s+eps)) is stable
            ratio_percentile: float = 99.0,  # keep top p% by ratio (ignored if topk used)
            min_target_percentile: float = 85.0,  # also require t_pool to be “high”
            topk: Optional[int] = None,  # OR: choose exactly top-k neurons per pair
            binary_weights: bool = False,  # True => intensity-free (1.0 on selected neurons)
            outlier_trim: float = 0.0,  # cosine-trim outlier pairs before aggregation
            aggregate: str = "pca",  # "pca" or "mean"
            tokens_for_target_pooling: str = "all",
            tokens_for_source_pooling: str = "all",
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
        hist = torch.zeros(65536).to(sae_dev)

        for src, tgt in sentence_pairs:
            # 1) get T5 embeddings exactly as generation uses
            h_src = pipeline._get_t5_prompt_embeds(prompt=[src], max_sequence_length=max_sequence_length)
            h_tgt = pipeline._get_t5_prompt_embeds(prompt=[tgt], max_sequence_length=max_sequence_length)

            # 2) SAE encode -> [1, T, D] -> squeeze token dim
            a_src = self.sae.encode(h_src.to(sae_dev)).squeeze(0)
            a_tgt = self.sae.encode(h_tgt.to(sae_dev)).squeeze(0)

            # Tokenize to get attention masks
            tok_src = self.t5_tokenizer(
                [src],
                max_length=max_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            )

            tok_tgt = self.t5_tokenizer(
                [tgt],
                max_length=max_sequence_length,
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
            # a_src = a_src[:L_src, :]   # [L_src, D]  <-- TRIMMED
            # a_tgt = a_tgt[:L_tgt, :]   # [L_tgt, D]

            src_tokens = self.get_tokens(src)[:L_src]
            tgt_tokens = self.get_tokens(tgt)[:L_tgt]

            # Source: indices of tokens to edit
            src_edit_idxs = []
            if tokens_for_source_pooling == "edit":
                src_edit_idxs, _ = self.get_idx_to_edit(src)

            # Target: indices that differ from source
            tgt_diff_idxs = []
            if tokens_for_target_pooling in ["diff", "edit,diff"]:
                j = 0
                for i, tok in enumerate(tgt_tokens):
                    if j < len(src_tokens) and tok == src_tokens[j]:
                        j += 1
                    else:
                        tgt_diff_idxs.append(i)

            # Target: indices of tokens in edit list
            tgt_edit_idxs = []
            if tokens_for_target_pooling in ["edit", "edit,diff"]:
                for i, tok in enumerate(tgt_tokens):
                    if tok in self.source_tokens_to_edit:
                        tgt_edit_idxs.append(i)

            # Final indices
            src_idxs = list(range(L_src)) if tokens_for_source_pooling == "all" else src_edit_idxs
            tgt_idxs = list(range(L_tgt)) if tokens_for_target_pooling == "all" else sorted(
                set(tgt_edit_idxs + tgt_diff_idxs))

            # 3) max-pool chosen tokens -> [D] # TODO: maybe we should use mean?
            # do math in float32 for numerical stability
            s_pool = a_src[src_idxs, :].max(dim=0).values.float()
            t_pool = a_tgt[tgt_idxs, :].max(dim=0).values.float()

            # 4) ratio score # TODO: intuituin?
            if use_log_ratio:
                ratio = torch.log((t_pool + eps) / (s_pool + eps))
            else:
                ratio = t_pool / (s_pool + eps)
                ratio /= ratio.max()

            # 5) build mask of "target-only" neurons
            if topk is not None:
                k = min(topk, ratio.numel())
                idx = torch.topk(ratio, k=k).indices
                mask = torch.zeros_like(ratio, dtype=torch.bool)
                mask[idx] = True
            else:
                r_thr = torch.quantile(ratio, ratio_percentile / 100.0)
                t_thr = torch.quantile(t_pool, min_target_percentile / 100.0)
                #                 print(ratio.topk(50))
                #                 mask = ratio >= 0.1
                mask = (ratio >= r_thr) & (t_pool >= t_thr)
                hist[mask] += 1
            #                 print(torch.where(mask != 0))

            if not mask.any():
                continue

            # 6) build the masked vector for this pair
            if binary_weights:
                v = torch.zeros_like(t_pool)
                v[mask] = 1.0
            else:
                # keep target activations on selected neurons if you want a tiny bit of weighting
                v = torch.zeros_like(t_pool)
                v[mask] = t_pool[mask] - s_pool[mask]  # TODO changed by me

            # normalize per-pair to avoid length bias
            # v_norm = v.norm(p=2)
            # if v_norm == 0 or not torch.isfinite(v_norm):
            #     continue
            # v = v / v_norm

            selected.append(v.unsqueeze(0))
        hist /= hist.max()
        #         print("histogram is: ", hist.topk(300))
        if not selected:
            raise ValueError("No pairs produced a usable activation mask. Relax thresholds or set topk.")

        M = torch.cat(selected, dim=0)  # [N_pairs, D]

        # 7) optional outlier trim by cosine similarity to mean
        if outlier_trim > 0 and M.size(0) >= 5:
            mean_dir = torch.nn.functional.normalize(M.mean(dim=0, keepdim=True), dim=1)
            cos = torch.nn.functional.cosine_similarity(M.float(), mean_dir.float(), dim=1)
            keep = torch.topk(cos, k=max(3, int((1.0 - outlier_trim) * M.size(0)))).indices
            M = M[keep]

        # 8) aggregate across pairs
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

        # 9) unit-norm & cast back to SAE dtype
        # g = g / (g.norm(p=2) + 1e-8)
        self.global_dir_sae = g.to(next(self.sae.parameters()).dtype).detach()
        #         print(hist.topk(100))
        #         self.global_dir_sae[hist <= 0.25] = 0
        return self.global_dir_sae

    # ---------- interpolate: use global dir ----------

    @torch.inference_mode()
    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        if self.global_dir_sae is None:
            raise RuntimeError(
                "Global direction not set. Call set_global_direction_from_pairs(...) first "
                "or pass global_dir_sae in __init__."
            )
        prompt_embeds = callback_kwargs["prompt_embeds"]
        prompt = callback_kwargs['prompt']
        a_src = self.get_sentence_sae_embed(prompt_embeds, prompt)
        idxs, _ = self.get_idx_to_edit(prompt)
        if idxs:
            g = self.global_dir_sae.to(a_src.device, dtype=a_src.dtype)
            step = self.interpolation_factors[step_index]
            for i in idxs:
                a_src[0, i, :] = a_src[0, i, :] + step * g
        return self.sae.decode(a_src).to(prompt_embeds.dtype)




