# SAEdit: Token-level control for continuous image editing via Sparse AutoEncoder
> **Ronen Kamenetsky, Sara Dorfman, Daniel Garibi, Roni Paiss, Or Patashnik, Daniel Cohen-Or**
>
> Large-scale text-to-image diffusion models have become the backbone of modern image editing, yet text prompts alone do not offer adequate control over the editing process. Two properties are especially desirable: disentanglement, where changing one attribute does not unintentionally alter others, and continuous control, where the strength of an edit can be smoothly adjusted. We introduce a method for disentangled and continuous editing through token-level manipulation of text embeddings. The edits are applied by manipulating the embeddings along carefully chosen directions, which control the strength of the target attribute. To identify such directions, we employ a Sparse Autoencoder (SAE), whose sparse latent space exposes semantically isolated dimensions. Our method operates directly on text embeddings without modifying the diffusion process, making it model agnostic and broadly applicable to various image synthesis backbones. Experiments show that it enables intuitive and efficient manipulations with continuous control across diverse attributes and domains.

<a href="https://ronen94.github.io/SAEdit/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
<a href="https://arxiv.org/abs/2510.05081"><img src="https://img.shields.io/badge/arXiv-SAEdit-b31b1b.svg" height=20.5></a>

<p align="center">
<img src="docs/teaser.jpg" width="800px"/>
</p>

## Description
The official implementation of the paper SAEdit: Token-level control for continuous image editing via Sparse AutoEncoder

## Environment Setup
Our code builds on the requirement of the `diffusers` library. To set up the environment, please run:
```
conda env create -f environment.yaml
conda activate saeedit_env
```
or install requirements:
```
pip install -r requirements.txt
```

## Usage
To run our method, insert the SAEdit block as a callback function as can be seen bellow, and in the example subfolder.
```
import torch
import yaml
from diffusers import FluxPipeline
from saedit import SAEditCallback

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = "a portrait of a man riding a donkey in the snow"
    sentence_pair_paths = "configs/variations/smiling_man.yaml"
    tokens_to_edit = ['man']

    # Load model
    model = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to(device)
    
    # Load SAE model
    sae = GlobalBatchTopKMatryoshkaSAE.from_pretrained(
                                        "Ronenk94/T5_matryoshka_sae",
                                         device="cuda")    
    # Load variation configuration
    with open(sentence_pair_paths, "r") as f:
        sentence_pairs = yaml.safe_load(f)
    
    # Create SAEdit callback
    callback = SAEditCallback(
        pipeline=model,
        sae=sae,
        prompt=prompt,
        source_tokens_to_edit=tokens_to_edit,
        factor=0.8, # set 0 to retrieve original image
        sentence_pairs=sentence_pairs["sentence_pairs"],
        max_sequence_length=256
    )
    
    # Generate image
    output = model(
        prompt=prompt,
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
```

## Training
We have made the [weights](https://huggingface.co/Ronenk94/T5_matryoshka_sae_top_300) of our model public in hugging face.

To train a new SAE model two steps are required:

### Step 1: Generate Embedding Dataset

First, generate a Zarr file containing T5 embeddings from your training dataset:

```bash
python src/train/generate_data.py --dataset <your_dataset> --output_path ./data/t5_embeddings.zarr
```

This will create a Zarr file containing all T5 embeddings from your training dataset.

### Step 2: Train the SAE

Train the SAE model using the generated embeddings:

```bash
python src/train/train.py --cfg src/train/configs/train_matryoshka_topk_300_dict_65k.yml
```

**Configuration Setup:**
- Update the `dataset_path` in your chosen config file (`.yml`) to point to the Zarr file created in Step 1
- Adjust other hyperparameters (batch size, learning rate, etc.) as needed
- The trained model will be saved as a snapshot (`.pt` file) that can be loaded and used with SAEdit



## Acknowledgements 
This code builds on the code from the [diffusers](https://github.com/huggingface/diffusers) library. In addition for defining and training the SAE code was borrowed from [Matryoshka_sae](https://github.com/bartbussmann/matryoshka_sae)

## Citation
If you use this code for your research, please cite the following work: 
```
@misc{kamenetsky2025saedittokenlevelcontrolcontinuous,
      title={SAEdit: Token-level control for continuous image editing via Sparse AutoEncoder}, 
      author={Ronen Kamenetsky and Sara Dorfman and Daniel Garibi and Roni Paiss and Or Patashnik and Daniel Cohen-Or},
      year={2025},
      eprint={2510.05081},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2510.05081}, 
}
```