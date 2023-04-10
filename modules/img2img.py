import utils
from PIL import Image
import random
import os

# 图生图
def img2img(model_name, init_image, prompt, negative_prompt, sampler, Image_size, strength, guidance_scale, num_inference_steps, seed):

    width, height = utils.get_size(Image_size)
    seed = random.randint(0, 2 ** 32) if seed == '-1' else int(seed)

    Image.fromarray(init_image).save("/home/aistudio/PPdiffusers-webui/temp_img.jpg")
    img2img = utils.img2img(
        pipe=utils.pipe,
        image_path="/home/aistudio/PPdiffusers-webui/temp_img.jpg",
        prompt=prompt,
        negative_prompt=negative_prompt,
        scheduler_name=sampler,
        width=width,
        height=height,
        strength=float(strength),
        num_inference_steps=min(int(num_inference_steps), 100),
        guidance_scale=float(guidance_scale),
        max_embeddings_multiples=3,
        enable_parsing=True,
        seed=seed,
        fp16=False)

    save_path = os.path.join("/home/aistudio/PPdiffusers-webui/output", "result_img2img_temp.jpg")
    img2img.save(save_path)
    return Image.open(save_path)