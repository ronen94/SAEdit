import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
import yaml
import time
import argparse
from src.models.sparse_autoencoders.matryoshka_sae import GlobalBatchTopKMatryoshkaSAE
from src.models.sparse_autoencoders.sae_factory import SAEFactory
import random
import numpy as np
from src.train.data import ZarrDataset
import wandb

wandb.login(key=os.getenv("WANDB_API_KEY"))
class MatryoshkaTrainer:
    def __init__(self, model: torch.nn.Module,
                 train_data: Dataset,
                 val_data: Dataset,
                 cfg, device: str="cuda"):
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = device
        self.model = model.to(self.device)
        self.data_loader_workers = cfg['data_loader_workers']
        self.train_data = self.prepare_dataloader(train_data, cfg['batch_size'])
        self.val_data = self.prepare_dataloader(val_data, cfg['batch_size'])
        self.save_every = cfg['checkpoint_freq']
        self.log_every = cfg['perf_log_freq']
        full_result_path = os.path.join(cfg['results_path'], cfg['name'])
        os.makedirs(full_result_path, exist_ok=True)
        self.snapshot_path = os.path.join(full_result_path, "snapshot.pt")
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
        self.cur_iteration = 0
        self.iter_ct = 0
        if os.path.exists(self.snapshot_path):
            print(f"Loading snapshot from: {self.snapshot_path}")
            self._load_snapshot(self.snapshot_path)



    def _run_batch(self, inputs):
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        loss = outputs['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["max_grad_norm"])
        self.model.make_decoder_weights_and_grad_unit_norm()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return outputs

    def _save_snapshot(self, cur_iteration, snapshot_path=None):
        if snapshot_path is None:
            snapshot_path = self.snapshot_path
        snapshot = {
            'model_state_dict': self.model.state_dict(),
            'cur_iteration': cur_iteration,
            'optimizer': self.optimizer.state_dict(),
            'iter_ct' : self.iter_ct
        }
        torch.save(snapshot, snapshot_path)

    def _load_snapshot(self, snapshot_path):
        if os.path.exists(snapshot_path):
            snapshot = torch.load(snapshot_path, map_location='cpu')
            self.model.load_state_dict(snapshot['model_state_dict'])
            self.cur_iteration = snapshot['cur_iteration']
            self.optimizer.load_state_dict(snapshot['optimizer'])
            if 'iter_ct' in snapshot:
                self.iter_ct = snapshot['iter_ct']
            print(f"Loaded snapshot from {snapshot_path}, starting from iteration {self.cur_iteration}.")
        else:
            print(f"No snapshot found at {snapshot_path}. Starting fresh.")

    def train_log(self, outputs, iter_ct, epoch, time):
        # Where the magic happens
        loss = outputs["loss"].item()
        wandb.log({"epoch": epoch, "loss": loss}, step=iter_ct)
        print(f"Loss after {str(iter_ct)} iterations: {loss}. Time from last log {time}", flush=True)
        if "l2_loss" in outputs:
            l2_loss = outputs["l2_loss"].item()
            wandb.log({"epoch": epoch, "l2_loss" : l2_loss}, step=iter_ct)
        if "entropy_loss" in outputs:
            entropy_loss = outputs["entropy_loss"].item()
            wandb.log({"epoch": epoch, "entropy_loss" : entropy_loss}, step=iter_ct)


    def get_validation_loss(self):
        total_loss = 0.0
        with torch.inference_mode():
            for batch in self.val_data:
                inputs = batch.to(self.device)
                outputs = self.model(inputs)
                total_loss += (outputs['sae_out'] - inputs).pow(2).mean()
        avg_loss = total_loss / len(self.val_data)
        return avg_loss

    def train(self, config):
        cur_time = time.time()
        wandb.watch(self.model, self.optimizer, log="all", log_freq=100)
        epoch = 0
        while epoch < config["num_epochs"]:
            dataset = create_dataset(dataset_path=config['dataset_path'], cache_size=config.train_cache_size)
            train_dataset = Subset(dataset, range(self.cur_iteration * config.batch_size, len(dataset) - config.val_data_size))
            self.train_data = self.prepare_dataloader(train_dataset, self.cfg['batch_size'])
            epoch += 1
            for i, batch in enumerate(self.train_data):
                self.iter_ct += 1
                outputs = self._run_batch(inputs=batch)
                if self.iter_ct % self.log_every == 0:
                    new_cur_time = time.time()
                    self.train_log(outputs, iter_ct=self.iter_ct, epoch=epoch, time=new_cur_time - cur_time)
                    cur_time = new_cur_time
                if self.iter_ct % self.save_every == 0 and i > 0:
                    self.cur_iteration = i
                    self._save_snapshot(i)
                    val_loss = self.get_validation_loss()
                    wandb.log({"validation_loss": val_loss}, step=self.iter_ct)
                    torch.cuda.empty_cache()
                if self.iter_ct % 3000 == 0 and self.iter_ct > 0:
                    saved_snapshot_path = self.snapshot_path.replace(".pt", f"_{i}.pt")
                    self._save_snapshot(self.iter_ct, snapshot_path=saved_snapshot_path)


    def prepare_dataloader(self, dataset: Dataset, batch_size: int):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,  # Disable shuffle for faster startup - data is already diverse
            num_workers= self.data_loader_workers,
            persistent_workers=True
        )


def create_dataset(dataset_path: str, max_samples: int = None, cache_size: int = 8_388_608):
    return ZarrDataset(dataset_path, max_samples=max_samples, cache_size=cache_size)



def main(train_cfg_path: str):
    with open(train_cfg_path, 'r') as f:
        train_cfg = yaml.safe_load(f)
    with wandb.init(config=train_cfg, project=train_cfg.wandb_name):
        config = wandb.config
        # Use max_samples and cache_size from config for testing/debugging
        max_samples = config.get('max_samples', None)
        cache_size = config.get('cache_size', train_cfg.train_cache_size)  # Default 8m samples per cache batch
        validation_data_size = config.get('validation_data_size', train_cfg.val_data_size)
        dataset = create_dataset(dataset_path=config['dataset_path'], max_samples=max_samples, cache_size=cache_size)
        train_dataset = Subset(dataset, range(0, len(dataset) - validation_data_size))
        validation_dataset = Subset(dataset, range(len(dataset) - validation_data_size, len(dataset)))
        model = SAEFactory().create_sae(config['sae_type'], config)
        model.train()
        trainer = MatryoshkaTrainer(model=model, train_data=train_dataset, val_data=validation_dataset, cfg=config)
        trainer.train(config)


if __name__ == "__main__":
    print(f'CUDA available: {torch.cuda.is_available()}')
    parser = argparse.ArgumentParser(description='matryoshka sae training job')
    parser.add_argument('--cfg', type=str, required=True, help='Path to the training configuration file')
    args = parser.parse_args()
    main(train_cfg_path=args.cfg)

