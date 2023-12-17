from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)

pipeline
i=0
items=[]
while i<3:
    item = input("Enter prompt:")
    items.append(item)
    i+=1

print(items)
for item in items:
    prompt = "Cartoon {}.".format(item)
    print(prompt)
    image = pipeline(prompt, num_inference_steps=15).images[0]

    image.save("image_{}.png".format(item))


print("generation completed!")