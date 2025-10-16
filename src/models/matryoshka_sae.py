
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from src.models.sae_base import BaseAutoencoder
import torch
import random
import numpy as np
import os



class GlobalBatchTopKMatryoshkaSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(cfg['seed'])
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        total_dict_size = sum(cfg["group_sizes"])
        self.group_sizes = cfg["group_sizes"]

        self.group_indices = [0] + list(torch.cumsum(torch.tensor(cfg["group_sizes"]), dim=0))

        self.active_groups = len(cfg["group_sizes"])

        self.b_dec = nn.Parameter(torch.zeros(self.config["act_size"]))
        self.b_enc = nn.Parameter(torch.zeros(total_dict_size))

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(cfg["act_size"], total_dict_size)
            )
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(total_dict_size, cfg["act_size"])
            )
        )

        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.num_batches_not_active = torch.zeros(total_dict_size, device=device)
        self.register_buffer('threshold', torch.tensor(0.0))

        # self.to(cfg["dtype"]).to(cfg["device"])

    def compute_activations(self, x_cent):
        pre_acts = x_cent @ self.W_enc
        acts = F.relu(pre_acts)

        if self.training:
            acts_topk = torch.topk(
                acts.flatten(),
                self.config["top_k"] * x_cent.shape[0],
                dim=-1
            )
            acts_topk = (
                torch.zeros_like(acts.flatten())
                .scatter(-1, acts_topk.indices, acts_topk.values)
                .reshape(acts.shape)
            )
            self.update_threshold(acts_topk)
        else:
            acts_topk = torch.where(acts > self.threshold, acts, torch.zeros_like(acts))

        return acts, acts_topk

    def encode(self, x):
        original_shape = x.shape
        x, x_mean, x_std = self.preprocess_input(x)
        self.x_mean = x_mean
        self.x_std = x_std

        x = x.reshape(-1, x.shape[-1])
        x_cent = x - self.b_dec
        _, result = self.compute_activations(x_cent)
        max_act_index = self.group_indices[self.active_groups]
        result[:, max_act_index:] = 0
        if len(original_shape) == 3:
            result = result.reshape(original_shape[0], original_shape[1], -1)
        return result

    def decode(self, acts_topk):
        reconstruct = acts_topk @ self.W_dec + self.b_dec
        return self.postprocess_output(reconstruct, self.x_mean, self.x_std)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        x_reconstruct = self.b_dec

        intermediate_reconstructs = []
        all_acts, all_acts_topk = self.compute_activations(x_cent)

        for i in range(self.active_groups):
            start_idx = self.group_indices[i]
            end_idx = self.group_indices[i + 1]
            W_dec_slice = self.W_dec[start_idx:end_idx, :]
            acts_topk = all_acts_topk[:, start_idx:end_idx]
            x_reconstruct = acts_topk @ W_dec_slice + x_reconstruct
            intermediate_reconstructs.append(x_reconstruct)

        self.update_inactive_features(all_acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, all_acts, all_acts_topk, x_mean,
                                    x_std, intermediate_reconstructs)
        return output

    def get_loss_dict(self, x, x_reconstruct, all_acts, all_acts_topk, x_mean, x_std, intermediate_reconstructs):
        total_l2_loss = (self.b_dec - x.float()).pow(2).mean()
        l2_losses = torch.tensor([]).to(x.device)
        for intermediate_reconstruct in intermediate_reconstructs:
            l2_losses = torch.cat([l2_losses, (intermediate_reconstruct.float() -
                                               x.float()).pow(2).mean().unsqueeze(0)])
            total_l2_loss += (intermediate_reconstruct.float() - x.float()).pow(2).mean()

        min_l2_loss = l2_losses.min()
        max_l2_loss = l2_losses.max()
        mean_l2_loss = total_l2_loss / (len(intermediate_reconstructs) + 1)

        l1_norm = all_acts_topk.float().abs().sum(-1).mean()
        l0_norm = (all_acts_topk > 0).float().sum(-1).mean()
        l1_loss = self.config["l1_coeff"] * l1_norm
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, all_acts)
        loss = mean_l2_loss + l1_loss + aux_loss

        num_dead_features = (self.num_batches_not_active > self.config["n_batches_to_dead"]).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": all_acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": mean_l2_loss,
            "min_l2_loss": min_l2_loss,
            "max_l2_loss": max_l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "threshold": self.threshold,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, all_acts):
        residual = x.float() - x_reconstruct.float()
        aux_reconstruct = torch.zeros_like(residual)

        acts = all_acts
        dead_features = self.num_batches_not_active >= self.config["n_batches_to_dead"]

        if dead_features.sum() > 0:
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.config["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            aux_reconstruct = aux_reconstruct + x_reconstruct_aux

        if aux_reconstruct.abs().sum() > 0:
            aux_loss = self.config["aux_penalty"] * (aux_reconstruct.float() - residual.float()).pow(2).mean()
            return aux_loss

        return torch.tensor(0.0, device=x.device)

    @torch.no_grad()
    def update_threshold(self, acts_topk, lr=0.01):
        positive_mask = acts_topk > 0
        if positive_mask.any():
            min_positive = acts_topk[positive_mask].min()
            self.threshold = (1 - lr) * self.threshold + lr * min_positive