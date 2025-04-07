# benchmarking

# 🔍 Gemma-Bench: A Benchmarking Suite for Evaluating Gemma and Other Open LLMs

Welcome to **Gemma-Bench** — a reproducible, extensible, and automated benchmarking framework for evaluating **Gemma models** on academic and custom datasets, and comparing them against other popular open-source models like **LLaMA 2**, **Mistral**, and **Falcon**.

> 🧪 This is a proof-of-concept for a full-scale GSoC 2025 project proposal. Contributions, feedback, and ideas are welcome!

---

## 🚀 Goals

- Benchmark **Gemma model variants** (2B, 7B, etc.) on standard academic datasets (MMLU, GSM8K, ARC, HellaSwag).
- Compare performance against other open LLMs (LLaMA 2, Mistral).
- Automate the evaluation process with modular scripts.
- Generate **visual reports** (charts, tables) and support **live leaderboard-style views**.
- Ensure **reproducibility** through Docker and simple CLI interfaces.
- Enable custom benchmarks (e.g., enterprise QA agent tasks inspired by real-world use cases).

---

## 🛠️ Features (Planned)

✅ Evaluate models on multiple-choice and generative tasks  
✅ Support for Gemma and Hugging Face-compatible LLMs  
✅ Clean output format (JSON/CSV)  
✅ Visualization via Matplotlib/Plotly  
✅ Modular dataset and model registration  
🚧 Leaderboard generation (coming soon)  
🚧 Web UI / dashboard (optional)  


---

## 📊 Sample Output

| Model      | MMLU Accuracy | GSM8K EM | Inference Time |
|------------|----------------|----------|----------------|
| Gemma 2B   | 34.2%          | 42.5%    | 1.2s           |
| LLaMA 2 7B | 45.3%          | 54.1%    | 1.7s           |
| Mistral 7B | 48.1%          | 56.3%    | 1.5s           |

📈 More visual charts to come!

---

## 🧪 Quick Start

```bash
git clone https://github.com/shikharxyz/gemma-bench.git
cd gemma-bench
pip install -r requirements.txt

# Run MMLU benchmark for Gemma 2B
python run_mmlu.py --model gemma-2b

# Datasets
MMLU

GSM8K

ARC

HellaSwag

Custom QA Evaluation (WIP)

🔍 Reproducibility
This project will support:

Docker-based reproducible environments

Config-driven evaluation pipelines

Open data and clear evaluation logs

🌱 GSoC 2025 Context
This repo is part of my Google Summer of Code 2025 proposal titled:

"Benchmarking Gemma: A Reproducible and Scalable Suite for Evaluating Open Models"

Key goals:

Build a sustainable tool for the open-source community.

Contribute back to the Gemma ecosystem.

Help LLM researchers and developers compare models with confidence.

🙋‍♂️ About Me
I'm Shikhar, an MS student in IT & Analytics with 6+ years of industry experience as a Software Developer at Walmart. I've built LLM-based QA agents for enterprise use, and I'm passionate about bridging research and engineering in the LLM space.

Feel free to connect or collaborate!

🧠 License
MIT License. All datasets are subject to their respective licenses.
