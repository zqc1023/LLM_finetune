import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from datasets import Dataset
from peft import get_peft_model, LoraConfig
import json
from datetime import datetime
import random
import argparse
import logging
import sys


class LoggerWriter:
    """Redirect stdout/stderr to logging, flush immediately."""

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        message = message.rstrip()
        if message:
            self.logger.log(self.level, message)
            for handler in self.logger.handlers:
                handler.flush()  # 确保实时写入文件

    def flush(self):
        for handler in self.logger.handlers:
            handler.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Train DPO")
    parser.add_argument("--input", type=str, required=True, help="input_file")
    parser.add_argument("--model", type=str, help="model")
    return parser.parse_args()


# ------------------工具函数------------------
def create_path(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder {folder_path} Created")
    else:
        print(f"Folder {folder_path} Existed")


def setup_logger(output_dir):
    log_file = os.path.join(output_dir, "train.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清理已有 handler
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件 handler
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    # 终端 handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Logger initialized. Logs will be saved to {log_file}")

    # 将 stdout/stderr 重定向到 logger
    sys.stdout = LoggerWriter(logger, logging.INFO)
    # sys.stderr = LoggerWriter(logger, logging.ERROR)

    return logger


def load_json(file_path, max_samples=20000):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data[:max_samples]


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_dpo():
    args = parse_args()
    # 文件和输出路径
    set_seed(42)
    train_file = args.input
    example_data = load_json(train_file)

    # 划分训练集和验证集

    random.shuffle(example_data)
    split_idx = int(0.85 * len(example_data))
    train_data = example_data[:split_idx]
    val_data = example_data[split_idx:]

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # ------------------生成带时间的输出路径------------------
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"/home/zhangqianchi/homework/saves/{current_time}"
    create_path(output_dir)

    logging_dir = os.path.join(output_dir, "logs")
    create_path(logging_dir)

    setup_logger(output_dir)

    # 加载模型和 tokenizer
    model_name = args.model
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ref_model.eval()  # 确保不训练

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 定义 PEFT 配置
    peft_config = LoraConfig(
        r=128,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_alpha=64,
        lora_dropout=0,  # 设置为0以优化性能
        bias="none",  # 设置为"none"以优化性能
    )
    model = get_peft_model(model, peft_config)

    # # ------------------DPO 配置------------------
    # logging_dir = os.path.join(output_dir, "logs")  # TensorBoard 日志路径
    # create_path(logging_dir)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        logging_dir=logging_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=3e-6,
        warmup_ratio=0.1,
        logging_steps=1,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        seed=42,
        beta=0.3,
        max_length=2048,
        max_prompt_length=2048,
        max_grad_norm=1.0,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Training finished. TensorBoard logs are in {logging_dir}")
    print("Run: tensorboard --logdir {}".format(logging_dir))


if __name__ == "__main__":
    train_dpo()
