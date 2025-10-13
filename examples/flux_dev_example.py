from src.saedit import SAEditCallback
import numpy as np
import torch

factor = 0. # an edit strengh factor. factor should normally be between 0 and 1
target_edit_token = 'man'
variation_path = "src/variations/smiling_man.yaml"
source_sentence = "a portrait of a man riding a donkey in the snow"
seed = 42
with open(variation_path, "r") as f:
    variation_data = yaml.safe_load(f)
sentence_pairs=variation_data["sentence_pairs"]


t5_block = SAEditCallback(
    factor=factor,
    prompt=source_sentence,
    source_tokens_to_edit=target_edit_token,  # ["man"]
    sae=sae_model_baseline,
    pipeline=model,
    sentence_pairs = sentence_pairs
)


out = model(
    prompt=source_sentence,
    guidance_scale=3.5,
    height=1024, width=1024,
    num_inference_steps=40,
    max_sequence_length=256,
    generator=torch.Generator(device='cuda').manual_seed(seed),
    callback_on_step_end=t5_block,
    callback_on_step_end_tensor_inputs=CALLBACK_ON_STEP_END_INPUT_TENSORS
)
out.images[0]