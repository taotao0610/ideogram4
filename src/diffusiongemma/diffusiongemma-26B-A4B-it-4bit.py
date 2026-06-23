from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model, processor = load("mlx-community/diffusiongemma-26B-A4B-it-4bit")
config = load_config("mlx-community/diffusiongemma-26B-A4B-it-4bit")

image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
prompt = "Describe this image."

formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=1
)

output = generate(model, processor, formatted_prompt, image)
print(output)