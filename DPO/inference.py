import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import argparse
from metrics import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run QA with LLaMA model and permuted docs"
    )
    parser.add_argument("--input", type=str, required=True, help="input_file")
    parser.add_argument("--model", type=str, default="qwen", help="model_name")

    return parser.parse_args()


def load_qwen(model_name):
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    return model, tokenizer


def load_qa_documents(data_path):
    q, a, docs = [], [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                q.append(d["question"])
                a.append(d["answer"])
                docs.append(d["documents"])
            except Exception as e:
                print(f"Error: {e}")
    return q, a, docs


def get_answer(documents, question, model, tokenizer):
    final_docs = "\n".join(
        [f"document {j}: {doc}" for j, doc in enumerate(documents, 1)]
    )
    system_prompt = ""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Based on the provided reference documents, answer the following question:{question}\n{final_docs}？",
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32,
        temperature=0.01,  # 控制随机性（越低越确定）
        do_sample=True,  # 一定要加！否则temperature/top_p/top_k都不会生效
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # print("thinking content:", thinking_content)
    print("content:", content)
    return content


def main():
    args = parse_args()
    question, answer, documents = load_qa_documents(args.input)
    model, tokenizer = load_qwen(args.model)
    number = 0
    answer_set = []

    for i in tqdm(range(len(question))):
        ans = get_answer(
            documents=documents[i],
            question=question[i],
            model=model,
            tokenizer=tokenizer,
        )
        answer_set.append(ans.strip().rstrip("."))

        number += sub_exact_match(ans, answer[i])
        print(number / (i + 1))
        avg_f1, _ = dataset_level_f1(answer_set, answer[0 : len(answer_set)])
        print("F1: ", avg_f1)
    avg_score, _ = batch_sub_exact_match(answer_set, answer)
    print("EM:", avg_score)
    avg_f1, _ = dataset_level_f1(answer_set, answer)
    print("F1:", avg_f1)


if __name__ == "__main__":
    main()
