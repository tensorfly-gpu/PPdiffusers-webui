from modules.helper.lora_helper import load_torch
import paddle
import safetensors.numpy
import os
__all__ = ['convert_paddle_lora_to_safetensor_lora', 'convert_pytorch_lora_to_paddle_lora']

def convert_paddle_lora_to_safetensor_lora(paddle_file, safe_file=None):
    if not os.path.exists(paddle_file):
        print(f"{paddle_file} 文件不存在！")
        return 
    if safe_file is None:
        safe_file = paddle_file.replace("paddle_lora_weights.pdparams", "pytorch_lora_weights.safetensors")
        
    tensors = paddle.load(paddle_file)
    new_tensors = {}
    for k, v in tensors.items():
        new_tensors[k] = v.cpu().numpy().T
    safetensors.numpy.save_file(new_tensors, safe_file)
    print(f"文件已经保存到{safe_file}!")
    
def convert_pytorch_lora_to_paddle_lora(pytorch_file, paddle_file=None):
    if not os.path.exists(pytorch_file):
        print(f"{pytorch_file} 文件不存在！")
        return 
    if paddle_file is None:
        paddle_file = pytorch_file.replace("pytorch_lora_weights.bin", "paddle_lora_weights.pdparams")
        
    tensors = load_torch(pytorch_file)
    new_tensors = {}
    for k, v in tensors.items():
        new_tensors[k] = v.T
    paddle.save(new_tensors, paddle_file)
    print(f"文件已经保存到{paddle_file}!")