# 环境配置，启动ui界面
import os
import webui

def prepare_environment():
    # 安装环境
    diffusers_auto_update()

def diffusers_auto_update():
    try:
        import safetensors
        from ppdiffusers.utils import image_grid
        from paddlenlp.transformers.clip.feature_extraction import CLIPFeatureExtractor
        from paddlenlp.transformers import FeatureExtractionMixin
        print("environment all ready!")

    except (ModuleNotFoundError, ImportError, AttributeError):
        print('检测到库不完整, 正在安装库')
        os.system("pip install -U pip  -i https://mirror.baidu.com/pypi/simple")
        os.system("pip install -U OmegaConf --user -i https://mirror.baidu.com/pypi/simple")
        os.system("pip install ppdiffusers==0.9.0 --user -i https://mirror.baidu.com/pypi/simple")
        os.system("pip install paddlenlp==2.4.9 --user -i https://mirror.baidu.com/pypi/simple")
        os.system("pip install -U safetensors --user -i https://mirror.baidu.com/pypi/simple")

def start():
    webui.demo.launch()

if __name__ == "__main__":
    prepare_environment()
    start()