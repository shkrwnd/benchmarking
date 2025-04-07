
import argparse
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

def load_mmlu_subset(subject="abstract_algebra", split="test", few_shot=False):
    dataset = load_dataset("hendrycks_test", subject, split=split)
    return dataset

def format_question(example):
    # Format the multiple choice question
    choices = ["A", "B", "C", "D"]
    prompt = f"Question: {example['question']}\n"
    for idx, choice in enumerate(example['choices']):
        prompt += f"{choices[idx]}. {choice}\n"
    prompt += "Answer:"
    return prompt

def main(model_name, subject):
    from datasets import get_dataset_config_names

    valid_subjects = get_dataset_config_names("hendrycks_test")
    if subject not in valid_subjects:
        raise ValueError(f"Subject '{subject}' is not valid. Choose from: {valid_subjects}")
    else:
        print("dataset present")
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if model.device.type == 'cuda' else -1)

    dataset = load_mmlu_subset(subject)
    correct = 0
    total = 0
    results = []

    print(f"Evaluating on subject: {subject}")
    for example in tqdm(dataset):
        prompt = format_question(example)
        output = generator(prompt, max_new_tokens=1, return_full_text=False)[0]["generated_text"].strip()
        predicted = output[0].upper() if output and output[0].upper() in ["A", "B", "C", "D"] else "Invalid"

        is_correct = predicted == example["answer"]
        correct += is_correct
        total += 1
        results.append({
            "question": example["question"],
            "predicted": predicted,
            "correct": example["answer"],
            "is_correct": is_correct
        })

    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}%")

    Path("results").mkdir(exist_ok=True)
    with open(f"results/mmlu_{subject.replace(' ', '_')}.json", "w") as f:
        json.dump({"accuracy": accuracy, "results": results}, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (Hugging Face ID)")
    parser.add_argument("--subject", type=str, default="abstract_algebra", help="MMLU subject")
    args = parser.parse_args()

    main(args.model, args.subject)
