# examples/basic_usage.py
"""
Basic example of using SAEdit for image editing.
"""
import torch
import yaml
from diffusers import FluxPipeline
from saedit import SAEditCallback

def main():
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to(device)
    
    # Load SAE model
    sae = torch.load("path/to/your/sae_model.pt")
    
    # Load variation configuration
    with open("configs/variations/smiling_man.yaml", "r") as f:
        variation_data = yaml.safe_load(f)
    
    # Create SAEdit callback
    callback = SAEditCallback(
        pipeline=model,
        sae=sae,
        source_tokens_to_edit=["man"],
        factor=0.8, # set 0 to retrieve original image
        sentence_pairs=variation_data["sentence_pairs"],
        prompt="a portrait of a man riding a donkey in the snow",
        max_sequence_length=256
    )
    
    # Generate image
    output = model(
        prompt="a portrait of a man riding a donkey in the snow",
        guidance_scale=3.5,
        height=1024,
        width=1024,
        num_inference_steps=40,
        max_sequence_length=256,
        generator=torch.Generator(device=device).manual_seed(42),
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=['prompt_embeds']
    )
    
    # Save result
    output.images[0].save("output.png")
    print("Image saved to output.png")

if __name__ == "__main__":
    main()