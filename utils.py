from contextlib import nullcontext
import paddle
from pipeline_stable_diffusion_all_in_one import StableDiffusionPipelineAllinOne
from PIL import Image
import cv2
import numpy as np

model_name = "Baitian/momocha"
# 初始化pipe。
pipe = StableDiffusionPipelineAllinOne.from_pretrained(model_name, safety_checker=None, feature_extractor=None,
                                                        requires_safety_checker=False)

support_scheduler = [
    "EulerAncestralDiscrete",
    "PNDM",
    "DDIM",
    "LMSDiscrete",
    "HeunDiscrete",
    "KDPM2AncestralDiscrete",
    "KDPM2Discrete"
]

model_name_list = [
    "Baitian/momocha",
    "Linaqruf/anything-v3.0",
    "MoososCap/NOVEL-MODEL",
    "Baitian/momoco",
    "hequanshaguo/monoko-e",
    "ruisi/anything",
    "hakurei/waifu-diffusion-v1-3",
    "CompVis/stable-diffusion-v1-4",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2",
    "stabilityai/stable-diffusion-2-base",
    "hakurei/waifu-diffusion",
    "naclbit/trinart_stable_diffusion_v2_60k",
    "naclbit/trinart_stable_diffusion_v2_95k",
    "naclbit/trinart_stable_diffusion_v2_115k",
    "ringhyacinth/nail-set-diffuser",
    "Deltaadams/Hentai-Diffusion",
    "BAAI/AltDiffusion",
    "BAAI/AltDiffusion-m9",
    "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1",
    "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1",
    "huawei-noah/Wukong-Huahua"]


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
    else:
        width, height = 512, 512
    return width, height

context_null = nullcontext()


def ReadImage(image, height=None, width=None):
    """
    Read an image and resize it to (height,width) if given.
    If (height,width) = (-1,-1), resize it so that
    it has w,h being multiples of 64 and in medium size.
    """
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')

    # clever auto inference of image size
    w, h = image.size
    if height == -1 or width == -1:
        if w > h:
            width = 768
            height = max(64, round(width / w * h / 64) * 64)
        else:  # w < h
            height = 768
            width = max(64, round(height / h * w / 64) * 64)
        if width > 576 and height > 576:
            width = 576
            height = 576
    if (height is not None) and (width is not None) and (w != width or h != height):
        image = image.resize((width, height), Image.ANTIALIAS)
    return image

def txt2img(pipe, prompt, scheduler_name, width, height, guidance_scale, num_inference_steps, negative_prompt,
            max_embeddings_multiples, enable_parsing, fp16=False, seed=None):
    scheduler = pipe.create_scheduler(scheduler_name)

    # 边运行的时候会边传递值到这里！
    def callback_fn(i, total, tqdm_progess):
        print(i, total, tqdm_progess.format_dict)

    if fp16 and scheduler_name != "LMSDiscrete":
        context = paddle.amp.auto_cast(True, level='O2')
    else:
        context = context_null
    with context:
        return pipe.text2image(
            prompt,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            max_embeddings_multiples=int(max_embeddings_multiples),
            skip_parsing=(not enable_parsing),
            scheduler=scheduler,
            callback=callback_fn,
        ).images[0]


def img2img(pipe, image_path, prompt, scheduler_name, height, width, strength, num_inference_steps, guidance_scale,
            negative_prompt, max_embeddings_multiples, enable_parsing, fp16=True, seed=None):
    scheduler = pipe.create_scheduler(scheduler_name)
    init_image = ReadImage(image_path, height=height, width=width)

    # 边运行的时候会边传递值到这里！
    def callback_fn(i, total, tqdm_progess):
        print(i, total, tqdm_progess.format_dict)

    if fp16 and scheduler_name != "LMSDiscrete":
        context = paddle.amp.auto_cast(True, level='O2')
    else:
        context = context_null
    with context:
        return pipe.img2img(prompt,
                            seed=seed,
                            image=init_image,
                            num_inference_steps=num_inference_steps,
                            strength=strength,
                            guidance_scale=guidance_scale,
                            negative_prompt=negative_prompt,
                            max_embeddings_multiples=int(max_embeddings_multiples),
                            skip_parsing=(not enable_parsing),
                            scheduler=scheduler,
                            callback=callback_fn
                            ).images[0]


def inpaint(pipe, image_path, mask_path, prompt, scheduler_name, height, width, num_inference_steps, strength,
            guidance_scale, negative_prompt, max_embeddings_multiples, enable_parsing, fp16=True, seed=None):
    scheduler = pipe.create_scheduler(scheduler_name)
    init_image = ReadImage(image_path, height=height, width=width)
    mask_image = ReadImage(mask_path, height=height, width=width)

    # 边运行的时候会边传递值到这里！
    def callback_fn(i, total, tqdm_progess):
        print(i, total, tqdm_progess.format_dict)

    if fp16 and scheduler_name != "LMSDiscrete":
        context = paddle.amp.auto_cast(True, level='O2')
    else:
        context = context_null
    with context:
        return pipe.inpaint(
            prompt,
            seed=seed,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            max_embeddings_multiples=int(max_embeddings_multiples),
            skip_parsing=(not enable_parsing),
            scheduler=scheduler,
            callback=callback_fn
        ).images[0]