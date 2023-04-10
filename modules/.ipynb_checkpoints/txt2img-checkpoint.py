import utils
from PIL import Image
import random
import os

# 文生图
def txt2img(model_name, prompt, negative_prompt, sampler, Image_size, guidance_scale, num_inference_steps, seed):

    width, height = utils.get_size(Image_size)
    seed = random.randint(0, 2 ** 32) if seed == '-1' else int(seed)

    txt2img = utils.txt2img(
        pipe=utils.pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        scheduler_name=sampler,
        width=width,
        height=height,
        guidance_scale=float(guidance_scale),
        num_inference_steps=min(int(num_inference_steps), 100),
        max_embeddings_multiples=3,
        enable_parsing=True,
        seed=seed)

    save_path = os.path.join("/home/aistudio/PPdiffusers-webui/output", "result_txt2img_temp.jpg")
    txt2img.save(save_path)
    return Image.open(save_path)