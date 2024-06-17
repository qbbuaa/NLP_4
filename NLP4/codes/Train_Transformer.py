import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer,TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import tensorflow as tf
import re

tf.config.set_visible_devices([], 'GPU')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
model = model.cuda()

file_path = "jyxstxtqj_downcc.com/三十三剑客图.txt"  # 替换为你的小说文件名
with open(file_path, "r", errors='ignore') as file:
    f = file.read()
    f = f.replace('\u3000', '')
    f = f.replace(' ', '')
    full_width_english = re.compile(r'[\uFF01-\uFF5E]+')
    novel_text = full_width_english.sub('', f)
# 将文本转换为GPT-2模型所需的格式
encoded_input = tokenizer.encode(novel_text, return_tensors="tf")

# 创建数据集
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=file_path,
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
model.save_pretrained('my_model')
tokenizer.save_pretrained('my_model')