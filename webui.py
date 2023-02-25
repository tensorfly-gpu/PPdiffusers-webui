# ui界面，调用后端推理

import argparse

parser = argparse.ArgumentParser()

cmd_opts = parser.parse_args()

def api_only():
    pass

def webui():
    pass

if __name__ == "__main__":
    if cmd_opts.nowebui:
        api_only()
    else:
        webui()