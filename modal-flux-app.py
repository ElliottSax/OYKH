"""
FLUX.1-dev on Modal - Production-Ready
60% cheaper than Replicate!
"""

import modal

stub = modal.Stub("oykh-flux-production")

flux_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "diffusers==0.30.3",
        "transformers",
        "accelerate",
        "safetensors",
        "torch",
        "sentencepiece",
        "protobuf"
    )
)

with flux_image.imports():
    import torch
    from diffusers import FluxPipeline

@stub.function(
    gpu="A10G",
    image=flux_image,
    timeout=600,
    container_idle_timeout=300
)
def generate_image(prompt: str, num_steps: int = 28, guidance: float = 3.5):
    """Generate image with FLUX.1-dev"""

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cuda")

    image = pipe(
        prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        height=1024,
        width=1024
    ).images[0]

    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

@stub.function(
    gpu="A10G",
    image=flux_image,
    timeout=600
)
def generate_with_lora(
    prompt: str,
    lora_path: str,
    num_steps: int = 28
):
    """Generate with custom LoRA"""

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    )

    pipe.load_lora_weights(lora_path)
    pipe = pipe.to("cuda")

    image = pipe(
        prompt,
        num_inference_steps=num_steps,
        guidance_scale=3.5,
        height=1024,
        width=1024
    ).images[0]

    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

@stub.webhook(method="POST")
def api(data: dict):
    """
    Public API endpoint for image generation

    POST https://your-username--oykh-flux-production-api.modal.run
    Body: {
        "prompt": "OYKHCHAR character celebrating",
        "use_lora": true,
        "num_steps": 28
    }
    """

    prompt = data.get("prompt")
    use_lora = data.get("use_lora", False)
    num_steps = data.get("num_steps", 28)

    if use_lora:
        image_bytes = generate_with_lora.remote(
            prompt,
            "your-username/oykhchar-lora",
            num_steps
        )
    else:
        image_bytes = generate_image.remote(prompt, num_steps)

    import base64
    return {
        "image": base64.b64encode(image_bytes).decode(),
        "format": "png"
    }

@stub.local_entrypoint()
def main(prompt: str = "A beautiful sunset"):
    """Test generation locally"""
    print(f"Generating: {prompt}")
    image_bytes = generate_image.remote(prompt)

    with open("output.png", "wb") as f:
        f.write(image_bytes)

    print("✅ Saved to output.png")
