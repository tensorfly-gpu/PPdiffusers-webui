import utils
from PIL import Image
import random
import os
import numpy as np
import cv2
import imageio

# inpaint图片后处理
def inpaint_post_process(image_path, mask_path, inpaint_path, save_path):
    ori_image = Image.open(image_path)
    mask = Image.open(mask_path)
    inpaint = Image.open(inpaint_path)

    ori_array = np.array(ori_image)[:, :, :3].transpose([2, 0, 1])
    C, H, W = ori_array.shape

    mask_array = np.array(mask)[:, :, :3] / 255
    mask_array = cv2.resize(mask_array, (W, H)).transpose([2, 0, 1])

    inpaint_array = np.array(inpaint)[:, :, :3]
    inpaint_array = cv2.resize(inpaint_array, (W, H)).transpose([2, 0, 1])

    result_array = mask_array * inpaint_array + (1 - mask_array) * ori_array

    result_array = result_array.transpose([1, 2, 0])
    result_array = result_array.astype(np.uint8)
    result_image = Image.fromarray(result_array)

    result_image.save(save_path)

# 局部重绘
def inpaint(model_name, init_image_mask, prompt, negative_prompt, sampler, Image_size, strength, guidance_scale, num_inference_steps, seed):
    width, height = utils.get_size(Image_size)
    seed = random.randint(0, 2 ** 32) if seed == '-1' else int(seed)

    imageio.imwrite("temp_img.png", init_image_mask["image"])
    imageio.imwrite("temp_mask_img.png", init_image_mask["mask"])

    inpaint = utils.inpaint(
        pipe=utils.pipe,
        image_path="temp_img.png",
        mask_path="temp_mask_img.png",
        prompt=prompt,
        negative_prompt=negative_prompt,
        scheduler_name=sampler,
        width=width,
        height=height,
        strength=float(strength),
        num_inference_steps=min(int(num_inference_steps), 100),
        guidance_scale=float(guidance_scale),
        max_embeddings_multiples=3,
        enable_parsing=True,    # 括号加强权重
        seed=seed,
        fp16=False)     #  半精度推理

    inpaint_save_path = os.path.join("output", "result_inpaint_temp.jpg")
    inpaint.save(inpaint_save_path)

    save_path = os.path.join("output", "result_inpaint.jpg")

    # 后处理inpaint图，非mask区域与原图保持一致
    inpaint_post_process(image_path="temp_img.png", mask_path="temp_mask_img.png", inpaint_path=inpaint_save_path,
                               save_path=save_path)

    return Image.open(save_path)