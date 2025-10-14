import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import os

class BaseAutoencoder(nn.Module):
    """Base class for autoencoder models."""

    def __init__(self, cfg):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = cfg
        torch.manual_seed(self.config["seed"])

        self.b_dec = nn.Parameter(torch.zeros(self.config["act_size"]))
        self.b_enc = nn.Parameter(torch.zeros(self.config["dict_size"]))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.config["act_size"], self.config["dict_size"])
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.config["dict_size"], self.config["act_size"])
            )
        )
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros((self.config["dict_size"],)).to(
            device
        )

    def preprocess_input(self, x):
        if self.config["input_unit_norm"]:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        if self.config["input_unit_norm"]:
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    @torch.no_grad()
    def update_inactive_features(self, acts):
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0

    def encode(self, x):
        raise NotImplementedError("Encode method must be implemented by subclasses")

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        import json
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config, f)

    @classmethod
    def from_pretrained(cls, load_directory, device="cuda", cache_dir=None):
        import json
        import os
        
        # Check if it's a local path or a HF repo
        if os.path.isdir(load_directory):
            # Local directory
            local_path = load_directory
        else:
            # Assume it's a HuggingFace repo ID
            try:
                from huggingface_hub import snapshot_download
                print(f"Downloading from HuggingFace: {load_directory}")
                local_path = snapshot_download(
                    repo_id=load_directory,
                    cache_dir=cache_dir
                )
                local_path = os.path.join(local_path, 'matryoshka_sae_top_300' )
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required to load from HuggingFace. "
                    "Install it with: pip install huggingface_hub"
                )
            except Exception as e:
                raise ValueError(
                    f"Could not load from '{load_directory}'. "
                    f"Please provide a valid local path or HuggingFace repo ID. "
                    f"Error: {e}"
                )
        
        # Load config
        config_path = os.path.join(local_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Initialize model
        model = cls(config)
        
        # Load weights
        model_path = os.path.join(local_path, "pytorch_model.bin")
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint)
        
        # Move to device
        model.to(device)
        model.eval()
        
        return model