#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import random
import gradio as gr
import os
import utils
from PIL import Image
from modules import txt2img, img2img, inpaint
import zipfile

## UI设计 ##
with gr.Blocks() as demo:
    # 顶部文字
    gr.Markdown("""
    # AI绘画
    ### 基于百度飞桨的大型开源项目：https://github.com/tensorfly-gpu/PPdiffusers-webui
    """)

    # 暂不支持换模型
    with gr.Row():
        with gr.Column():
            model_name = gr.Dropdown(utils.model_name_list, label="请选择一个基础模型", default="Baitian/momocha", multiselect=False)    # 同时作为训练lora的预训练模型
        with gr.Column():
            lora_name = gr.Dropdown(utils.model_name_list, label="请选择一个lora模型", default="text_encoder_unet_lora.safetensors", multiselect=False)
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
                        img2img_image_init = gr.Image(label="请上传图片")

                    # 生成、重置按钮（row：行）
                    with gr.Row():
                        img2img_button = gr.Button("生成")
                        # clear_button = gr.Button("重置")

                with gr.Column():  # 右边一列是输出
                    # 输出框
                    img2img_output = gr.Image(type="pil")

        with gr.TabItem("局部重绘"):
            # 一行 两列 左边一列是输入 右边一列是输出
            with gr.Row():
                with gr.Column():  # 左边一列是输入

                    inpaint_prompt = gr.Textbox(label="prompt", lines=3, placeholder="请输入正面描述", interactive=True,
                                        value=None)
                    inpaint_negative_prompt = gr.Textbox(label="negative_prompt", lines=2, placeholder="请输入负面描述",
                                                 interactive=True, value=None)

                    with gr.Row():
                        inpaint_sampler = gr.Dropdown(utils.support_scheduler, label="Sampling method", default="DDIM",
                                              multiselect=False)
                        inpaint_steps = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Sampling steps",
                                          interactive=True)

                    with gr.Row():
                        inpaint_Image_size = gr.Dropdown(["512x512", "512x768", "768x512", "640x640", "自动判断"], default="512x768",
                                                 label="Image size", multiselect=False)
                        # 返回只能显示一张图片, 待修复
                        # 先用只能一次生成一张图片替代
                        inpaint_num_images = 1
                        # num_images = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Num images", interactive=True)

                        inpaint_seed = gr.Textbox(label="seed", default="-1", lines=1, placeholder="请输入种子，默认-1")

                    with gr.Row():
                        inpaint_cfg_scale = gr.Slider(minimum=1, maximum=20, value=7, step=0.1, label="CFG Scale", interactive=True)
                        inpaint_strength = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="strength",
                                          interactive=True)

                    # 上传图片，生成图片
                    with gr.Row():
                        inpaint_image_mask_init = gr.Image(tool="sketch", label="请上传图片，然后使用鼠标涂抹",
                                                           type="numpy")

                with gr.Column():  # 右边一列是输出
                    # 输出框
                    inpaint_output = gr.Image(type="pil")
                    inpaint_button = gr.Button("生成")

        with gr.TabItem("训练"):
            # 多个tab
            with gr.Tabs():
                # 训练第一种lora
                with gr.TabItem("train dreambooth lora"):
                    # 使用1行2列
                    with gr.Row():
                        with gr.Column():  # 左边一列是输入
                            # dataset通过解压上传的压缩包上传时，同时启动训练

                            # TODO: 其他参数设置

                            file_upload = gr.File()
                            output_text = gr.Textbox()

                            train_dreambooth_lora_button = gr.Button("开始训练")

                        with gr.Column():  # 右边一列是输出
                            # 输出框
                            test2 = gr.Textbox(label="seed", default="-1", lines=1, placeholder="请输入种子，默认-1")

                with gr.TabItem("train text2image lora"):
                    test3 = gr.Textbox(label="seed", default="-1", lines=1, placeholder="请输入种子，默认-1")

    txt2img_button.click(fn=txt2img,
                      inputs=[model_name, txt2img_prompt, txt2img_negative_prompt, txt2img_sampler, txt2img_Image_size, txt2img_cfg_scale, txt2img_steps, txt2img_seed],
                      outputs=txt2img_output)

    img2img_button.click(fn=img2img,
                         inputs=[model_name, img2img_image_init, img2img_prompt, img2img_negative_prompt, img2img_sampler, img2img_Image_size, img2img_strength, img2img_cfg_scale, img2img_steps, img2img_seed],
                         outputs=img2img_output)

    inpaint_button.click(fn=inpaint,
                         inputs=[model_name, inpaint_image_mask_init, inpaint_prompt, inpaint_negative_prompt, inpaint_sampler, inpaint_Image_size, inpaint_strength, inpaint_cfg_scale, inpaint_steps, inpaint_seed],
                         outputs=inpaint_output)

    train_dreambooth_lora_button.click(
        fn=utils.train_dreambooth_lora,
        inputs=file_upload,
        outputs=output_text)

