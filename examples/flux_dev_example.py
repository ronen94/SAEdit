# examples/basic_usage.py
"""
Basic example of using SAEdit for image editing.
"""
import torch
import yaml
import argparse
from diffusers import FluxPipeline
from saedit import SAEditCallback

def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="SAEdit image generation with configurable parameters")
    parser.add_argument("--variation_path", type=str, required=True, 
                       help="Path to the variation YAML configuration file")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for image generation")
    parser.add_argument("--factor", type=float, default=0.,
                       help="SAEdit factor (set 0 to retrieve original image)")
    parser.add_argument("--source_tokens", type=str, nargs='+', required=True,
                       help="Source tokens to edit (space-separated list)")
    parser.add_argument("--output", type=str, default="output.png",
                       help="Output image filename (default: output.png)")
    
    return parser.parse_args()

def main(args):
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to(device)
    
    # Load SAE model
    sae = GlobalBatchTopKMatryoshkaSAE.from_pretrained(
                                        "Ronenk94/T5_matryoshka_sae",
                                         device="cuda")
    
    # Load variation configuration
    with open(args.variation_path, "r") as f:
        variation_data = yaml.safe_load(f)
    
    # Create SAEdit callback
    callback = SAEditCallback(
        pipeline=model,
        sae=sae,
        source_tokens_to_edit=args.source_tokens,
        factor=args.factor,
        sentence_pairs=variation_data["sentence_pairs"],
        prompt=args.prompt,
        max_sequence_length=256
    )
    
    # Generate image
    output = model(
        prompt=args.prompt,
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
    output.images[0].save(args.output)
    print(f"Image saved to {args.output}")

if __name__ == "__main__":
    args = parse_args()
    main(args)