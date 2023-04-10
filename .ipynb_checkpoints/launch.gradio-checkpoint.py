#该应用创建工具共包含三个区域，顶部工具栏，左侧代码区，右侧交互效果区，其中右侧交互效果是通过左侧代码生成的，存在对照关系。
#顶部工具栏：运行、保存、新开浏览器打开、实时预览开关，针对运行和在浏览器打开选项进行重要说明：
#[运行]：交互效果并非实时更新，代码变更后，需点击运行按钮获得最新交互效果。
#[在浏览器打开]：新建页面查看交互效果。
#以下为应用创建工具的示例代码

import gradio as gr

def quickstart(name):
    return "欢迎使用BML CodeLab应用创建工具Gradio, " + name + "!!!"
demo = gr.Interface(fn=quickstart, inputs="text", outputs="text")

demo.launch()
