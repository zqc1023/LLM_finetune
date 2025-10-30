import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


path1 = "/home/zhangqianchi/homework/dataset/train_res1.json"
path2 = "/home/zhangqianchi/homework/dataset/train_res2.json"
path3 = "/home/zhangqianchi/homework/dataset/train_res3.json"
path4 = "/home/zhangqianchi/homework/dataset/train_res4.json"


def load_data():
    final_data = []
    for path in [path1, path2, path3, path4]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            final_data.extend(data)
    return final_data


def load_qwen(model_name="/home/zhangqianchi/homework/model/Qwen3-8B-AWQ"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    return model, tokenizer


def main():
    data = load_data()
    _, tokenizer = load_qwen()
    system_prompt = "Based on the provided reference documents, answer the following question:{question}\n{document}？"
    final_dpo_data = []

    for item in tqdm(data):
        question = item["question"]
        documents = item["documents"]
        answer = item["answer"]
        docs = "\n".join(documents)
        for ans in answer:
            if ans in docs:
                answer = ans
                break
        if type(answer) == list:
            continue

        prompt = system_prompt.format(question=question, document="\n".join(documents))
        messages = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": prompt,
            },
        ]
        prompt_str = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        final_dpo_data.append(
            {
                "prompt": prompt_str,
                "chosen": answer,
                "rejected": item["wrong_answer"],
            }
        )
        output_path = "/home/zhangqianchi/homework/dpo_data.json"

        with open(output_path, "w", encoding="utf-8") as f:
            for item in final_dpo_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
