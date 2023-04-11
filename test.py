import os
import random
import utils
from modules.helper import lora_helper
from contextlib import nullcontext

def del_file(file_path):
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(e)


def get_size(standard_size):
    if standard_size == '512x768':
        width, height = 512, 768
    elif standard_size == '768x512':
        width, height = 768, 512
    elif standard_size == '512x512':
        width, height = 512, 512
    elif standard_size == '640x640':
        width, height = 640, 640
    elif standard_size == '自动判断':
        width, height = -1, -1
    return width, height

# 初始化pipe。
pipe = utils.pipe

prompt = 'red skirt, Xinhai'
negative_prompt = 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'

standard_size = "640x640"
num_images = 5
sampler = "DDIM"
width, height = get_size(standard_size)
superres_model_name = "无"
num_inference_steps = 50
seed = "-1"

# image_path = 'resources/clockcat.jpg'
# mask_path = "resources/mask_skirt.jpg"

guidance_scale = 8.5
# strength = 0.7

# for i in range(num_images):
#     cur_seed = random.randint(0, 2**32) if seed == '-1' else seed
#     inpaint = utils.inpaint(
#         pipe=pipe,
#         image_path=image_path,
#         mask_path=mask_path,
#         prompt=prompt,
#         negative_prompt=negative_prompt,
#         scheduler_name=sampler,
#         width=width,
#         height=height,
#         strength=float(strength),
#         num_inference_steps=min(int(num_inference_steps), 100),
#         guidance_scale=float(guidance_scale),
#         max_embeddings_multiples=3,
#         enable_parsing=True,
#         seed=cur_seed,
#         fp16=False)
#     save_path = os.path.join("output", "result_temp.jpg")
#     inpaint.save(save_path)

for i in range(num_images):
    cur_seed = random.randint(0, 2**32) if seed == '-1' else seed
    inpaint = utils.txt2img(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        scheduler_name=sampler, # 这里设置采样器无效
        width=width,
        height=height,
        num_inference_steps=min(int(num_inference_steps), 100),
        guidance_scale=float(guidance_scale),
        max_embeddings_multiples=3,
        enable_parsing=True,
        seed=cur_seed,
        fp16=False)
    save_path = os.path.join("output", f"lora_Xinhai_test_{i}.jpg")
    inpaint.save(save_path)