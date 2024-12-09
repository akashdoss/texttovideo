from diffusers import DiffusionPipeline

# Load the CogVideoX-5b model
pipe = DiffusionPipeline.from_pretrained("THUDM/CogVideoX-5b")

# Define the prompt for the image generation
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# Generate the image based on the prompt
image = pipe(prompt).images[0]

# Save the image to a file
image.save("generated_image.png")

# Display a confirmation
print("Image generated and saved as 'generated_image.png'")
