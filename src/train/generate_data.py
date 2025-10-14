import os
import torch
import json
import zarr
import numpy as np
import time
import argparse
from torch.utils.data import DataLoader, Subset
from transformers import T5EncoderModel
import torch.distributed as dist
from src.train.trainer import create_dataset
import numcodecs
from torch.distributed import init_process_group, destroy_process_group
from zarr.codecs import BloscCodec, BloscShuffle


def ddp_setup():
    rank = int(os.environ["LOCAL_RANK"])
    init_process_group(backend="nccl")
    torch.cuda.set_device(rank)


@torch.inference_mode()
def get_t5_prompt_embeds(
        text_inputs=None,
        device="cuda",
        num_images_per_prompt: int = 1,
        return_attention_mask: bool = False,
        text_encoder: T5EncoderModel = None,
):
    text_input_ids = text_inputs.input_ids
    batch_size = text_input_ids.shape[0]
    text_input_ids = text_input_ids.squeeze(1)

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)[0]
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    if return_attention_mask:
        attention_mask = text_inputs.attention_mask.squeeze(1)
        return prompt_embeds, attention_mask
    return prompt_embeds

# Add argparse to allow text_encoder_path and output_base as script arguments

def parse_args():
    parser = argparse.ArgumentParser(description="Generate T5 embeddings in distributed mode")
    parser.add_argument("--text_encoder_path", type=str, required=True, help="Path to the pretrained T5 encoder")
    parser.add_argument("--output_base", type=str, required=True, help="Base output directory for embeddings")
    parser.add_argument("--data_output_len", type=int, default=800_000_000, help="The length of the dataset created")
    return parser.parse_args()

def main():
    # Get distributed training info from environment variables (set by torchrun)

    args = parse_args()
    ddp_setup()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # Config
    data_row_number = args.data_output_len
    text_encoder_path = args.text_encoder_path
    output_base = args.output_base
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    
    # Load model on the assigned GPU
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_path).to(device)
    text_encoder.eval()



    # Dataset indexing - each process handles a slice
    full_dataset = create_dataset(max_length=256)
    total_len = len(full_dataset)
    
    
    # Output path for this process
    output_path = f"{output_base}/t5_embedding_gpu.zarr"
    if not os.path.exists(output_path) and local_rank == 0:
            store = zarr.storage.LocalStore(output_path)
            root = zarr.group(store=store, overwrite=True)
            embeddings_zarr = root.create_array(
                "embeddings",
                shape=(data_row_number, 4096),
                chunks=(1000, 4096),
                dtype="f2",
                compressors=[BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)]

            )
            root.attrs["row_idx"] = 0
            root.attrs["cache_idx"] = 0

    dist.barrier()

    store = zarr.storage.LocalStore(output_path)
    root = zarr.open(store, mode="r+")
    embeddings_zarr = root["embeddings"]
    start_idx = root.attrs["row_idx"]

    indicies = list(range(start_idx + local_rank, total_len, world_size))
    cache_slice_size = data_row_number // world_size
    cur_cache_idx = root.attrs["cache_idx"] + cache_slice_size * local_rank
    cache_end_idx = data_row_number if local_rank == world_size - 1 else (local_rank + 1) * cache_slice_size


    
    if len(indicies) < 1 or cur_cache_idx >= cache_end_idx:
        print(f"[GPU {local_rank}] Already completed. Skipping.")
        return
    
    
    # Update start_idx for resuming
    start_idx = indicies[0]
    print(f"[GPU {local_rank}] Resuming from row {start_idx}/{total_len}")

    
    # Create dataset subset for this process
    dataset = Subset(full_dataset, indices=indicies)
    print(f"[GPU {local_rank}] Starting with {len(dataset)} samples -> {output_path}")
    
    # DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=50,
        pin_memory=True,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )
    
    # Embedding loop
    start_time = time.time()
    batch_idx = 0
    
    for batch in data_loader:
        batch_idx += 1
        sentence_number = batch.input_ids.shape[0]
        batch, attention = get_t5_prompt_embeds(
            text_inputs=batch, 
            text_encoder=text_encoder, 
            device=device,
            return_attention_mask=True
        )
        
        bool_mask = attention.bool()
        batch = batch[bool_mask]
        batch = batch.reshape(-1, 4096)
        rows_to_add = batch.shape[0]
        
        if cur_cache_idx + rows_to_add >= cache_end_idx:
            rows_to_add = cache_end_idx - cur_cache_idx
            batch = batch[:rows_to_add]
        
        if rows_to_add <= 0:
            break
            
        embeddings_zarr[cur_cache_idx:cur_cache_idx + rows_to_add, :] = batch.cpu().numpy()
        if local_rank == 0:
            root.attrs["row_idx"] += sentence_number
            root.attrs["cache_idx"] += rows_to_add
        cur_cache_idx += rows_to_add


        
        if batch_idx % 100 == 0:
            print(f"[GPU {local_rank}] {cur_cache_idx} rows processed in {time.time() - start_time:.1f}s", flush=True)
            start_time = time.time()
    
    print(f"[GPU {local_rank}] Finished -> {output_path}, number of rows filled is {cur_cache_idx}")
    destroy_process_group()


if __name__ == "__main__":
    main()