
import argparse
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import re
from pathlib import Path

def normalize_answer(answer):
    """Extract the final numeric answer from a generated string."""
    match = re.search(r'(-?\\d+(?:\\.\\d+)?)', answer)
    return match.group(1).strip() if match else None

def main(model_name, max_samples=100):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if model.device.type == 'cuda' else -1)

    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test[:{}]".format(max_samples))

    correct = 0
    total = 0
    results = []

    for example in tqdm(dataset):
        prompt = f"Question: {example['question']}\nAnswer:"
        output = generator(prompt, max_new_tokens=100, return_full_text=False)[0]["generated_text"]
        predicted = normalize_answer(output)
        expected = normalize_answer(example["answer"])

        is_correct = predicted == expected
        correct += is_correct
        total += 1

        results.append({
            "question": example["question"],
            "predicted": predicted,
            "expected": expected,
            "is_correct": is_correct
        })

    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}%")

    Path("results").mkdir(exist_ok=True)
    with open("results/gsm8k_results.json", "w") as f:
        json.dump({"accuracy": accuracy, "results": results}, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (Hugging Face ID)")
    parser.add_argument("--max_samples", type=int, default=100, help="Number of test examples to evaluate")
    args = parser.parse_args()

    main(args.model, args.max_samples)
