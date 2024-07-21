import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import gradio as gr
from transformers import *

# 通过 Interface 加载 pipeline 并启动文本分类任务
#gr.Interface.from_pipeline(pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese")).launch()

# 通过 Interface 加载 pipeline 并启动阅读理解任务
gr.Interface.from_pipeline(pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa")).launch()