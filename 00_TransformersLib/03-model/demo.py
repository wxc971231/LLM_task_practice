import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(base_path)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam

class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv(f"{base_path}/ChnSentiCorp_htl_all.csv")    # 加载原始数据
        self.data = self.data.dropna()                                      # 去掉 nan 值

    def __getitem__(self, index):
        text:str = self.data.iloc[index]["review"]
        label:int = self.data.iloc[index]["label"]
        return text, label
    
    def __len__(self):
        return len(self.data)

def collate_func(batch):
    # 对 dataloader 得到的 batch data 进行后处理
    # batch data 是一个 list，其中每个元素是 (sample, label) 形式的元组
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    
    # 对原始 texts 列表进行批量 tokenize，通过填充或截断保持 token 长度为 128，要求返回的每个字段都是 pytorch tensor
    global tokenizer
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")

    # 增加 label 字段
    inputs["labels"] = torch.tensor(labels)
    return inputs

def evaluate(model):
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    return acc_num / len(validset)

def train(model, optimizer, epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch) # batch 是一个字典，其中包含 model forward 方法所需的字段，每个字段 value 是 batch tensor
            output.loss.backward()  # batch 字典中包含 labels 时会计算损失，详见源码
            optimizer.step()
            if global_step % log_step == 0:
                print(f"ep: {ep}, global_step: {global_step}, loss: {output.loss.item()}")
            global_step += 1
        acc = evaluate(model)
        print(f"ep: {ep}, acc: {acc}")

if __name__ == "__main__":
    # 构造训练集/测试集以及对应的 Dataloader
    dataset = MyDataset()
    train_size = int(0.9*len(dataset))
    vaild_size = len(dataset) - train_size
    trainset, validset = random_split(dataset, lengths=[train_size, vaild_size])
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate_func)
    validloader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=collate_func)

    # 构造 tokenizer、model 和 optimizer
    tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
    model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")  # 从 AutoModelForSequenceClassification 加载标准初始化模型，从 AutoModel.from_pretrained("hfl/rbt3") 加载 ckpt 权重模型
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = Adam(model.parameters(), lr=2e-5)

    # 训练
    train(model, optimizer)

    # 测试
    sen = "我觉得这家酒店不错，饭很好吃！"
    id2_label = {0: "差评！", 1: "好评！"}
    model.eval()
    with torch.inference_mode():
        inputs = tokenizer(sen, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1)
        print(f"输入：{sen}\n模型预测结果:{id2_label.get(pred.item())}")
