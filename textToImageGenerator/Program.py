from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)

pipeline
prompt = "A photograph of an astronaut riding a horse on Mars, high resolution, high definition."
image = pipeline(prompt, num_inference_steps=10).images[0]
image
image.save("image_of_squirrel_painting.png")