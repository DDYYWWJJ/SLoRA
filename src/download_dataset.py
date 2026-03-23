from datasets import load_dataset
import os

# 改用 HTTP 代理（确保你的代理软件开启了 HTTP 端口，通常是 7890）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'  # 注意改成 http://
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["WANDB_DISABLED"] = "true"

dataset = load_dataset("openai_humaneval", download_mode="force_redownload")
print('over openai_humaneval')
dataset = load_dataset("gsm8k", "main", download_mode="force_redownload")  # 数据集名称和子集参考[1,7](@ref)
print('over gsm8k')
dataset = load_dataset("zwhe99/commonsense_170k", download_mode="force_redownload")
print('over commonsense_170k')
