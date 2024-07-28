import os
base_path = os.path.abspath(os.path.join(os.path.dirname('__file__')))

import torch
import numpy as np
from transformers import Trainer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from typing import Dict, List
from transformers import AutoConfig
from transformers import AutoTokenizer, LlamaTokenizerFast
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

def small_init_weights(model):
    ''' 
        SAMLLINIT是Xavier初始化的变体，这里还结合了GPT2的初始化方法
        参考：https://zhuanlan.zhihu.com/p/676464982
    ''' 
    for name, param in model.named_parameters():
        if 'embed_tokens.weight' in name:
            param.data.normal_(mean=0.0, std=model.config.initializer_range)
        elif 'self_attn' in name:
            param.data.normal_(mean=0.0, std=np.sqrt(2/(model.config.hidden_size + model.config.intermediate_size)))
        elif 'down_proj' in name:
            param.data.normal_(mean=0.0, std=np.sqrt(2/(model.config.hidden_size + model.config.intermediate_size)) / np.sqrt(config.num_hidden_layers))
        elif 'bias' in name:
            torch.nn.init.constant_(param, 0)

def kaiming_initialization(model):
    ''' 
        凯明初始化，更适合类 ReLU 激活
        参考 https://zhuanlan.zhihu.com/p/305055975
    '''
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            torch.nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='leaky_relu')
        elif 'bias' in name:
            torch.nn.init.constant_(param, 0)

def sample_preprocess(examples: Dict[str, List], tokenizer) -> Dict[str, List]:
    ''' 对数据样本进行 Tokenize 处理，输入输出都是 Dict[str, List] 格式以支持 batch 形式处理 '''
    # 得到 token 序列最大长度，根据原始数据可视化图，选择 2048 基本不会浪费数据
    max_token = 2048

    # 进行 tokenize，跳过所有 special_tokens，在 LLaMA 中就是不会在句首加上 bos_token <s>
    encoded_texts = tokenizer(examples['text'], add_special_tokens=False)   
    
    # 保留后 <=max_token-1 token，再加上 eos_token，总长度 <= max_token
    # 从保留尾部而非头部序列是重要的，因为数据中必须包含 eos_token，这样模型才能学会何时结束输出
    new_input_ids_list, new_attn_mask_list = [], []
    for input_ids in encoded_texts['input_ids']:
        temp = input_ids[-max_token+1:] + [tokenizer.eos_token_id]
        new_input_ids_list.append(temp)
        new_attn_mask_list.append([1] * len(temp))  # 所有 token 都可以关注，attention 全 1

    return {
        "input_ids": new_input_ids_list,
        "attention_mask": new_attn_mask_list
    }

def inference(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, input_text: str = "Once upon a time, ", max_new_tokens: int = 16):
    ''' 使用模型进行推理生成 '''
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=40,
        top_p=0.95,
        temperature=0.8
    )
    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    return generated_text

if __name__ == "__main__":
    # ------------------------------------------------- 数据加载和预处理 -----------------------------------------------------
    # 加载原始数据集
    train_dataset = load_dataset("noanabeshima/TinyStoriesV2", split='train[:10%]', cache_dir='E:\8. data\hugggingface')
    val_dataset = load_dataset("noanabeshima/TinyStoriesV2", split='validation', cache_dir='E:\8. data\hugggingface')

    # 使用 LLama2 的词表（32k）进行 tokenize，LLama3 的太大了（128k）
    tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')
    train_dataset = train_dataset.shuffle()         # 训练集打乱一下
    tokenized_train_dataset = train_dataset.map(
        lambda example: sample_preprocess(example, tokenizer),
        batched=True,
        batch_size=10000,
        remove_columns=train_dataset.column_names,  # 移除原始数据字段
        desc='Running tokenizer on train_set: '
    )
    tokenized_val_dataset = val_dataset.map(
        lambda example: sample_preprocess(example, tokenizer),
        batched=True,
        batch_size=10000,
        remove_columns=val_dataset.column_names,    # 移除原始数据字段
        desc='Running tokenizer on val_set: '
    )

    # DataCollator 类似 torch Dataloader 中的 collect_fn，将 Dataset 中的原始数据组合成 batch，得到可以直接输入模型的形式
    # 为了组成 batch，DataCollator 通常会应用某些处理（比如padding到相同长度），有些还会在数据批次上应用随机数据增强（比如随机masking）
    # DataCollatorForLanguageModeling 在设置 mlm=False 时，直接适用于自回归 FLM 训练目标
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    # ---------------------------------------------------- 创建模型 -------------------------------------------------------
    # 根据 TinyStory 论文设置模型超参数
    hidden_size = 256                                               
    num_hidden_layers = 4
    intermediate_size = (int(hidden_size * 8/3 / 128) + 1) * 128    # 中间层取 8/3 倍，按 128 向上取整。用 8/3 而非 2 倍是因为 SwiGLU 包含三个参数矩阵   
    num_attention_heads = 16                                        # 注意力头共 16 个
    num_key_value_heads = 8                                         # 注意力头分 8 组，每组内 2 个头共用 key 和 value，即 GQA 机制
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 改动 llama 中的以上超参数默认值，其余保持不变
    config = AutoConfig.for_model(
        model_type="llama",
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,    
        num_attention_heads=num_attention_heads,                 
        num_key_value_heads=num_key_value_heads                   
    )

    # 从 config 加载模型，不使用预训练 ckpt，序列生成类模型要选择 CausalLM，使用 float32 全精度参数
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32).to(device)


    # ----------------------------------------------- 创建Trainer并启动训练 ---------------------------------------------------
    os.environ['WANDB_DISABLED'] = 'true'   # 关闭 wandb

    # 创建训练配置
    training_args = TrainingArguments(
        output_dir=f'{base_path}/01_TinyStory/output',      # 输出路径，包括模型检查点、中间文件等
        save_total_limit=2,                                 # output_dir 内留存的检查点最大数目
        overwrite_output_dir=True,                          # 是否覆写 output_dir
        num_train_epochs=2,                                 # 训练轮数，2 ~ 3 即可
        per_device_train_batch_size=2,                      # 训练 batch_size（一个 epoch 存在 ceil(data_num/bsz) 个 batch）
        per_device_eval_batch_size=5,                      # 验证 batch_size
        gradient_accumulation_steps=1,                      # 梯度累计步大小，显存不变时扩展等效 batch_size，小模型没必要。参考 https://blog.csdn.net/wxc971231/article/details/139177793
        do_train=True,                                      # 是否做训练
        do_eval=True,                                       # 是否做评估
        evaluation_strategy='steps',                        # 使用 steps 策略时，eval_steps 才有效（这是默认设置）
        eval_steps=1000,                                    # 评估步骤间隔
        save_strategy='steps',                              # 使用 steps 策略时，save_steps 才有效（这是默认设置）
        save_steps=1000,                                    # 检查点保存步骤间隔
        logging_steps=50,                                   # 打印步骤间隔
        learning_rate=1e-4,                                 # 学习率大小
        lr_scheduler_type='cosine',                         # 学习率调度策略，LLM 训练一般都用余弦
        bf16=torch.cuda.is_bf16_supported(),                # 尝试配置 bf16
        fp16=not torch.cuda.is_bf16_supported(),            # bf16 不行就上 fp16
        report_to=None,                                     # 日志输出目标，不想用 wandb 可以设置为 None
        seed=3407                                           # 随机种子
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,                    # 模型实例
        args=training_args,             # 训练参数
        train_dataset=train_dataset,    # 训练集
        eval_dataset=val_dataset,       # 验证集（评估集）
        data_collator=data_collator,    # data collator
    )

    # 开始训练
    trainer.train()

    # ----------------------------------------------- 训练结束，测试推理 ---------------------------------------------------
    sentance = inference(
        model,
        tokenizer,
        "Once upon a time, in a beautiful garden, there lived a little rabbit named Peter Rabbit.",
        max_new_tokens=256
    )
    print(sentance)

    # 将训练好的模型保存到本地
    model.save_pretrained(f'{base_path}/01_TinyStory/model/')