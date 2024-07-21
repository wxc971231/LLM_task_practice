from transformers import AutoTokenizer
from datasets import *

# 数据映射功能主要是结合 tokenizer 使用的，方便进行数据预处理
def preprocess_function(example, tokenizer):
    model_inputs = tokenizer(example["content"], max_length=512, truncation=True)
    labels = tokenizer(example["title"], max_length=32, truncation=True)
    model_inputs["labels"] = labels["input_ids"]    # 摘要任务，title 的编码结果作为 label
    return model_inputs

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    datasets = load_dataset("madao33/new-title-chinese")

    # 调用 .map() 方法时设置 num_proc=n 启动多进程处理
    # 当处理方法 preprocess_function 内含有不支持 batch 的方法时，还可以用多进程加速
    preprocess_func = lambda exp, tnier=tokenizer: preprocess_function(exp, tnier)
    processed_dataset = datasets.map(preprocess_func, num_proc=4)
    print(processed_dataset)
