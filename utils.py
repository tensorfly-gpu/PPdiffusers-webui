from contextlib import nullcontext
from pipeline import StableDiffusionPipelineAllinOne
from ppdiffusers import DPMSolverMultistepScheduler
from PIL import Image
import cv2
import numpy as np
from modules.helper import lora_helper
import paddle
import os
import zipfile

# 基础模型，需要是paddle版本的权重，未来会加更多的权重
pretrained_model_name_or_path = r"C:\Users\Administrator\.cache\paddlenlp\ppdiffusers\runwayml\stable-diffusion-v1-5"
# 我们加载safetensor版本的权重
lora_outputs_path = "text_encoder_unet_lora.safetensors"
# 加载之前的模型
pipe = StableDiffusionPipelineAllinOne.from_pretrained(pretrained_model_name_or_path, safety_checker=None, feature_extractor=None,
                                                        requires_safety_checker=False)
# 设置采样器，采样器移到这里实现
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# 加载lora权重, 可以选择加载和不加载lora, 没有lora时注释下行
# pipe.apply_lora(lora_outputs_path)
# pipe.apply_lora()

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
    # scheduler = pipe.create_scheduler(scheduler_name)

    # 边运行的时候会边传递值到这里！
    def callback_fn(i, total, tqdm_progess):
        print(i, total)
        # print(i, total, tqdm_progess.format_dict)

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
            # scheduler=scheduler,
            callback=callback_fn,
        ).images[0]


def img2img(pipe, image_path, prompt, scheduler_name, height, width, strength, num_inference_steps, guidance_scale,
            negative_prompt, max_embeddings_multiples, enable_parsing, fp16=True, seed=None):
    # scheduler = pipe.create_scheduler(scheduler_name)
    init_image = ReadImage(image_path, height=height, width=width)

    # 边运行的时候会边传递值到这里！
    def callback_fn(i, total, tqdm_progess):
        print(i, total)
        # print(i, total, tqdm_progess.format_dict)

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
                            # scheduler=scheduler,
                            callback=callback_fn
                            ).images[0]


def inpaint(pipe, image_path, mask_path, prompt, scheduler_name, height, width, num_inference_steps, strength,
            guidance_scale, negative_prompt, max_embeddings_multiples, enable_parsing, fp16=True, seed=None):
    # scheduler = pipe.create_scheduler(scheduler_name)
    init_image = ReadImage(image_path, height=height, width=width)
    mask_image = ReadImage(mask_path, height=height, width=width)

    # 边运行的时候会边传递值到这里！
    def callback_fn(i, total, tqdm_progess):
        print(i, total)
        # print(i, total, tqdm_progess.format_dict)

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
            # scheduler=scheduler,
            callback=callback_fn
        ).images[0]

# train dreambooth lora, aistudio project: https://aistudio.baidu.com/aistudio/projectdetail/5481677
# TODO 在训练中可视化输出的图到UI界面，拖动滑块查看不同时间步产生的结果
def train_dreambooth_lora(zip_file, pretrained_model_name_or_path="Baitian/momocha", instance_data_dir="./Xinhai", output_dir="./dream_booth_lora_outputs", instance_prompt="Xinhai", resolution=512, train_batch_size=1,
                          gradient_accumulation_steps=1, checkpointing_steps=50, learning_rate=1e-4, report_to="visualdl", lr_scheduler="constant", lr_warmup_steps=0,
                          max_train_steps=100, lora_rank=128, validation_prompt="Xinhai", validation_epochs=25, validation_guidance_scale=5.0, use_lion=False, seed=0):
    def unzip_file(zip_file):
        os.makedirs("train_dreambooth_lora", exist_ok=True)
        with zipfile.ZipFile(zip_file) as zip_ref:
            zip_ref.extractall('./train_dreambooth_lora')

    def process_zip_file(zip_file):
        unzip_file(zip_file.name)
        print("文件已解压并处理完成。")
        # 在这里添加您的处理代码
        os.system(f'python modules/train_dreambooth_lora.py \
          --pretrained_model_name_or_path={pretrained_model_name_or_path}  \
          --instance_data_dir="./train_dreambooth_lora" \
          --output_dir={output_dir} \
          --instance_prompt={instance_prompt} \
          --resolution={resolution} \
          --train_batch_size={train_batch_size} \
          --gradient_accumulation_steps={gradient_accumulation_steps} \
          --checkpointing_steps={checkpointing_steps} \
          --learning_rate={learning_rate} \
          --report_to={report_to} \
          --lr_scheduler={lr_scheduler} \
          --lr_warmup_steps={lr_warmup_steps} \
          --max_train_steps={max_train_steps} \
          --lora_rank={lora_rank} \
          --validation_prompt={validation_prompt} \
          --validation_epochs={validation_epochs} \
          --validation_guidance_scale={validation_guidance_scale} \
          --use_lion {use_lion} \
          --seed={seed}')
        return "训练完成！"

    return process_zip_file(zip_file)