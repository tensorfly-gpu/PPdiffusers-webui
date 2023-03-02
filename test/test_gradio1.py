import gradio as gr
import zipfile
import os

def unzip_file(zip_file):
    os.makedirs("unzipped", exist_ok=True)
    with zipfile.ZipFile(zip_file) as zip_ref:
        zip_ref.extractall('./unzipped')

def process_zip_file(zip_file):
    unzip_file(zip_file.name)
    # 在这里添加您的处理代码
    return "文件已解压并处理完成。"

with gr.Blocks() as demo:
    file_upload = gr.File()
    output_text = gr.Textbox()

    inpaint_button = gr.Button("生成")

    inpaint_button.click(
        fn=process_zip_file,
        inputs=file_upload,
        outputs=output_text)

demo.launch()