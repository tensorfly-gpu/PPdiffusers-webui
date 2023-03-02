import gradio as gr
import zipfile

def unzip_file(zip_file):
    with zipfile.ZipFile(zip_file) as zip_ref:
        zip_ref.extractall('./unzipped')

def process_zip_file(zip_file):
    unzip_file(zip_file.name)
    # 在这里添加您的处理代码
    return "文件已解压并处理完成。"

file_upload = gr.inputs.File(label="上传.zip文件")
output_text = gr.outputs.Textbox()

gr.Interface(
    process_zip_file,
    inputs=file_upload,
    outputs=output_text,
    title="上传和处理.zip文件",
    description="上传并处理.zip文件",
).launch()