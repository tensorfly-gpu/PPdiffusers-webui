# 环境配置，启动ui界面
import os

def prepare_environment():
    # 安装环境
    diffusers_auto_update()

def diffusers_auto_update():
    try:
        import safetensors, ppdiffusers, paddlenlp
        from ppdiffusers.utils import image_grid
        from paddlenlp.transformers.clip.feature_extraction import CLIPFeatureExtractor
        from paddlenlp.transformers import FeatureExtractionMixin
        if not (safetensors.__version__ == "0.2.8"
                and paddlenlp.__version__ == "2.5.1"
                and ppdiffusers.__version__ == "0.11.0"):
            raise ImportError
        print("environment all ready!")

    except (ModuleNotFoundError, ImportError, AttributeError):
        print('检测到库不完整或版本不正确, 正在安装库')
        # os.system("pip install -U gradio --user -i https://mirror.baidu.com/pypi/simple")
        os.system("pip install -U pip  -i https://mirror.baidu.com/pypi/simple")
        os.system("pip install -U OmegaConf --user -i https://mirror.baidu.com/pypi/simple")
        os.system("pip install ppdiffusers==0.11.0 --user -i https://mirror.baidu.com/pypi/simple")
        os.system("pip install paddlenlp==2.5.1 --user -i https://mirror.baidu.com/pypi/simple")
        os.system("pip install safetensors==0.2.8 --user -i https://mirror.baidu.com/pypi/simple")
        os.system("pip install paddlehub --user -i https://mirror.baidu.com/pypi/simple")
        # os.system("python -m pip install paddlepaddle-gpu==2.4.2.post112 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html --user")

def start():
    import webui
    webui.demo.launch(share=True)

if __name__ == "__main__":
    prepare_environment()
    start()
