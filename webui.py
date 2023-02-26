#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import random
import gradio as gr
import os
import utils
from PIL import Image

pipe = utils.StableDiffusionPipelineAllinOne.from_pretrained("Baitian/momocha", safety_checker=None, feature_extractor=None, requires_safety_checker=False)

# 文生图
def txt2img(model_name, prompt, negative_prompt, sampler, Image_size, guidance_scale, num_inference_steps, seed):
    # pipe = utils.StableDiffusionPipelineAllinOne.from_pretrained(model_name, safety_checker=None, feature_extractor=None, requires_safety_checker=False)

    width, height = utils.get_size(Image_size)
    seed = random.randint(0, 2 ** 32) if seed == '-1' else int(seed)

    txt2img = utils.txt2img(
        pipe=pipe,
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

    save_path = os.path.join("output", "result_txt2img_temp.jpg")
    txt2img.save(save_path)
    return Image.open(save_path)

# 图生图
def img2img(model_name, init_image, prompt, negative_prompt, sampler, Image_size, strength, guidance_scale, num_inference_steps, seed):
    # pipe = utils.StableDiffusionPipelineAllinOne.from_pretrained(model_name, safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    width, height = utils.get_size(Image_size)
    seed = random.randint(0, 2 ** 32) if seed == '-1' else int(seed)

    Image.fromarray(init_image).save("temp_img.jpg")
    img2img = utils.img2img(
        pipe=pipe,
        image_path="temp_img.jpg",
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

    save_path = os.path.join("output", "result_img2img_temp.jpg")
    img2img.save(save_path)
    return Image.open(save_path)

with gr.Blocks() as demo:
    # 顶部文字
    gr.Markdown("""
    # AI绘画
    ### 基于百度飞桨的大型开源项目：https://github.com/tensorfly-gpu/PPdiffusers-webui
    """)

    # 暂不支持换模型
    model_name = gr.Dropdown(utils.model_name_list, label="请选择一个基础模型", default="Baitian/momocha", multiselect=False)
    # 多个tab
    with gr.Tabs():
        with gr.TabItem("文生图"):
            # 一行 两列 左边一列是输入 右边一列是输出
            with gr.Row():
                with gr.Column():  # 左边一列是输入

                    txt2img_prompt = gr.Textbox(label="prompt", lines=3, placeholder="请输入正面描述", interactive=True, value=None)
                    txt2img_negative_prompt = gr.Textbox(label="negative_prompt", lines=2, placeholder="请输入负面描述", interactive=True, value=None)

                    with gr.Row():
                        txt2img_sampler = gr.Dropdown(utils.support_scheduler, label="Sampling method", default="DDIM", multiselect=False)
                        txt2img_steps = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Sampling steps", interactive=True)

                    with gr.Row():
                        txt2img_Image_size = gr.Dropdown(["512x512", "512x768", "768x512", "640x640"], default="512x768", label="Image size", multiselect=False)
                        # 返回只能显示一张图片, 待修复
                        # 先用只能一次生成一张图片替代
                        txt2img_num_images = 1
                        # num_images = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Num images", interactive=True)
                        txt2img_seed = gr.Textbox(label="seed", default="-1", lines=1, placeholder="请输入种子，默认-1")

                    txt2img_cfg_scale = gr.Slider(minimum=1, maximum=20, value=7, step=0.1, label="CFG Scale", interactive=True)

                    # 生成、重置按钮（row：行）
                    with gr.Row():
                        txt2img_button = gr.Button("生成")
                        # clear_button = gr.Button("重置")

                with gr.Column():  # 右边一列是输出
                    # 输出框
                    txt2img_output = gr.Image(type="pil")

        with gr.TabItem("图生图"):
            # 一行 两列 左边一列是输入 右边一列是输出
            with gr.Row():
                with gr.Column():  # 左边一列是输入

                    img2img_prompt = gr.Textbox(label="prompt", lines=3, placeholder="请输入正面描述", interactive=True,
                                        value=None)
                    img2img_negative_prompt = gr.Textbox(label="negative_prompt", lines=2, placeholder="请输入负面描述",
                                                 interactive=True, value=None)

                    with gr.Row():
                        img2img_sampler = gr.Dropdown(utils.support_scheduler, label="Sampling method", default="DDIM",
                                              multiselect=False)
                        img2img_steps = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Sampling steps",
                                          interactive=True)

                    with gr.Row():
                        img2img_Image_size = gr.Dropdown(["512x512", "512x768", "768x512", "640x640", "自动判断"], default="512x768",
                                                 label="Image size", multiselect=False)
                        # 返回只能显示一张图片, 待修复
                        # 先用只能一次生成一张图片替代
                        img2img_num_images = 1
                        # num_images = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Num images", interactive=True)

                        img2img_seed = gr.Textbox(label="seed", default="-1", lines=1, placeholder="请输入种子，默认-1")

                    with gr.Row():
                        img2img_cfg_scale = gr.Slider(minimum=1, maximum=20, value=7, step=0.1, label="CFG Scale", interactive=True)
                        img2img_strength = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="strength",
                                          interactive=True)
                        img2img_image_init = gr.Image(shape=(200, 200), label="请上传图片")

                    # 生成、重置按钮（row：行）
                    with gr.Row():
                        img2img_button = gr.Button("生成")
                        # clear_button = gr.Button("重置")

                with gr.Column():  # 右边一列是输出
                    # 输出框
                    img2img_output = gr.Image(type="pil")

    txt2img_button.click(fn=txt2img,
                      inputs=[model_name, txt2img_prompt, txt2img_negative_prompt, txt2img_sampler, txt2img_Image_size, txt2img_cfg_scale, txt2img_steps, txt2img_seed],
                      outputs=txt2img_output)

    img2img_button.click(fn=img2img,
                         inputs=[model_name, img2img_image_init, img2img_prompt, img2img_negative_prompt, img2img_sampler,
                                 img2img_Image_size, img2img_strength, img2img_cfg_scale, img2img_steps, img2img_seed],
                         outputs=img2img_output)

