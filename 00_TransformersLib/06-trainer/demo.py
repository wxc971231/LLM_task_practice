import os
base_path = os.path.abspath(os.path.join(os.path.dirname('__file__')))

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
from transformers import DataCollatorWithPadding

def sample_preprocess(examples, tokenizer):
    # 使用 tokenizer 处理样本的 review 字段，通过裁剪确保长度 <= 128
    tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
    # 在处理后的数据中增加 label 字段
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples

def eval_metric(eval_predict, acc_metric, f1_metric):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc

if __name__ == "__main__":
    # 加载数据集
    dataset = load_dataset("csv", data_files=f"{base_path}/datasets/ChnSentiCorp_htl_all.csv", split="train")
    dataset = dataset.filter(lambda x: x["review"] is not None) # 清洗 none 数据
    datasets = dataset.train_test_split(test_size=0.1)          # 划分训练集和测试集，得到 DatasetDict 对象

    # 数据批量预处理
    tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
    tokenized_datasets = datasets.map(
        lambda example: sample_preprocess(example, tokenizer),  # 对每个样本使用该方法处理
        batched=True,                                           # 使用 batch 形式加速
        batch_size=1000,                                        # 批处理尺寸
        remove_columns=datasets["train"].column_names           # 移除原始数据字段
    )

    # 创建模型
    model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")

    # 创建训练配置
    train_args = TrainingArguments(
        output_dir=f"{base_path}/00_TransformersLib/06-trainer/checkpoints", # 输出文件夹
        num_train_epochs=3,                                                 # 训练的 epoch 总数
        per_device_train_batch_size=64,                                     # 训练 batch_size（训练集共 6988 条数据，一个 epoch 存在 ceil(6988/64)=110 个 batch）
        per_device_eval_batch_size=128,                                     # 验证 batch_size
        logging_steps=10,                                                   # training log 打印的频率
        evaluation_strategy="epoch",                                        # 评估策略，每个 epoch 结束时评估
        save_strategy="epoch",                                              # 保存策略，每个 epoch 结束时保存
        save_total_limit=3,                                                 # 最大保存数，这样会保存最新的三个模型，之前的自动删掉
        learning_rate=2e-5,                                                 # 学习率
        weight_decay=0.01,                                                  # weight_decay
        metric_for_best_model="f1",                                         # 训练过程中，用这个指标确定历史最优模型
        load_best_model_at_end=True                                         # 训练完成后加载历史最优模型
    )

    # 创建评估函数
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    # 创建 Trainer
    trainer = Trainer(
        model=model, 
        args=train_args, 
        train_dataset=tokenized_datasets["train"], 
        eval_dataset=tokenized_datasets["test"], 
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),         # data_collator 就是 Dataloader 的 collect_fn，这里 DataCollatorWithPadding 会把每个 batch data padding 到该 batch 中最长的序列长度
        compute_metrics=lambda eval_predict: eval_metric(eval_predict, acc_metric, f1_metric)
    )

    # 开始训练
    trainer.train()

    # 评估训练好的模型（由于 TrainingArguments 中设置了 load_best_model_at_end=True，此时自动加载了历史最优模型）
    print(trainer.evaluate(tokenized_datasets["test"]))

    # 使用训练好的模型进行预测
    print(trainer.predict(tokenized_datasets["test"]))