#!/usr/bin/env python3
"""Standalone training script converted from Qwen3_(4B)_GRPO.ipynb."""

import argparse
import gc
import io
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

from unsloth import FastLanguageModel  # noqa: E402

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from safetensors import safe_open
from transformers import TextStreamer
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
from vllm import SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT + GRPO fine-tuning for Qwen3-4B.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where GRPO checkpoints and logs are stored.",
    )
    parser.add_argument(
        "--lora-output",
        default="grpo_saved_lora",
        help="Path to save the trained LoRA adapter.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a very small training loop for quick validation.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Force 4-bit base model loading to reduce VRAM.",
    )
    return parser.parse_args()


class Tee(io.TextIOBase):
    """Duplicate writes to multiple streams (console + log file)."""

    def __init__(self, *streams):
        super().__init__()
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def prepare_run_directory(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def init_run_logging(run_dir: Path):
    log_file = open(run_dir / "run.log", "w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)
    return log_file, original_stdout, original_stderr


def record_run_arguments(
    run_dir: Path, args: argparse.Namespace, extra: dict | None = None
) -> None:
    metadata = {
        "command": " ".join(sys.argv),
        "args": vars(args),
        "run_directory": str(run_dir),
    }
    if extra:
        metadata["extra_paths"] = extra
    with open(run_dir / "run_args.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = prepare_run_directory(Path(args.output_dir))
    log_file, original_stdout, original_stderr = init_run_logging(run_dir)
    trainer_output_dir = run_dir / "trainer_outputs"
    lora_output_path = run_dir / Path(args.lora_output).name
    trainer_output_dir.mkdir(parents=True, exist_ok=True)
    record_run_arguments(
        run_dir,
        args,
        extra={
            "trainer_output_dir": str(trainer_output_dir),
            "lora_output_dir": str(lora_output_path),
        },
    )
    print(f"Run artifacts will be stored in: {run_dir}")

    max_seq_length = 2048
    lora_rank = 8
    load_in_4bit = args.load_in_4bit
    gpu_memory_utilization = 0.75 
    conservativeness = 0.5
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        max_lora_rank=lora_rank,
        full_finetuning=False,
        gpu_memory_utilization=gpu_memory_utilization,
        conservativeness=conservativeness,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    reasoning_start = "<start_working_out>"
    reasoning_end = "<end_working_out>"
    solution_start = "<SOLUTION>"
    solution_end = "</SOLUTION>"

    system_prompt = (
        "You are given a problem.\n"
        "Think about the problem and provide your working out.\n"
        f"Place it between {reasoning_start} and {reasoning_end}.\n"
        f"Then, provide your solution between {solution_start}{solution_end}"
    )

    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + eos_token }}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        "{{ '" + system_prompt + "' + eos_token }}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '" + reasoning_start + "' }}"
        "{% endif %}"
    )
    tokenizer.chat_template = chat_template

    def format_dataset(row: pd.Series) -> list[dict[str, str]]:
        expected_answer = row["expected_answer"]
        problem = row["problem"]
        thoughts = row["generated_solution"].replace("<think>", "").replace("</think>", "")
        thoughts = thoughts.strip()
        final_prompt = (
            reasoning_start
            + thoughts
            + reasoning_end
            + solution_start
            + expected_answer
            + solution_end
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": final_prompt},
        ]

    dataset_df = load_dataset("unsloth/OpenMathReasoning-mini", split="cot").to_pandas()[
        ["expected_answer", "problem", "generated_solution"]
    ]
    numeric_mask = pd.to_numeric(pd.Series(dataset_df["expected_answer"]), errors="coerce").notnull()
    dataset_df = dataset_df.iloc[np.where(numeric_mask)[0]].copy()
    # if args.smoke_test:
    #     dataset_df = dataset_df.head(64).copy()

    dataset_df["Messages"] = dataset_df.apply(format_dataset, axis=1)
    dataset_df["N"] = dataset_df["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))
    filtered_df = dataset_df.loc[dataset_df["N"] <= max_seq_length / 2].copy()
    if filtered_df.empty:
        filtered_df = dataset_df.head(1).copy()

    filtered_df["text"] = tokenizer.apply_chat_template(
        filtered_df["Messages"].values.tolist(), tokenize=False
    )
    dataset = Dataset.from_pandas(filtered_df)

    sft_epochs = 1 if args.smoke_test else 2
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            warmup_ratio=0.03,
            num_train_epochs=sft_epochs,
            learning_rate=2e-4,
            logging_steps=1 if args.smoke_test else 5,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
            max_steps=100 if args.smoke_test else 10000,
            gradient_checkpointing=True
        ),
    )
    trainer.train()

    text = tokenizer.apply_chat_template(
        dataset[0]["Messages"][:2],
        tokenize=False,
        add_generation_prompt=True,
    )
    _ = model.generate(
        **tokenizer(text, return_tensors="pt").to(device),
        temperature=0,
        max_new_tokens=1024,
        streamer=TextStreamer(tokenizer, skip_prompt=False),
    )

    del dataset
    torch.cuda.empty_cache()
    gc.collect()

    dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    if args.smoke_test:
        dataset = dataset.select(range(min(len(dataset), 100)))

    def extract_hash_answer(text: str) -> str:
        return text

    dataset = dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["prompt"]},
            ],
            "answer": extract_hash_answer(x["solution"]),
        }
    )

    eos_token = tokenizer.eos_token or ""
    solution_end_regex = r"</SOLUTION>[\s]{0,}" + (f"(?:{re.escape(eos_token)})?" if eos_token else "")
    match_format = re.compile(
        rf"{reasoning_end}.*?"
        rf"{solution_start}(.+?){solution_end_regex}"
        rf"[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL,
    )

    def match_format_exactly(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            if match_format.search(response) is not None:
                score += 3.0
            scores.append(score)
        return scores

    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            score += 0.5 if response.count(reasoning_end) == 1 else -1.0
            score += 0.5 if response.count(solution_start) == 1 else -1.0
            score += 0.5 if response.count(solution_end) == 1 else -1.0
            scores.append(score)
        return scores

    def check_answer(prompts, completions, answer, **kwargs):
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [
            guess.group(1) if (guess := match_format.search(r)) is not None else None
            for r in responses
        ]
        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0
            if guess is None:
                scores.append(-2.0)
                continue
            if guess == true_answer:
                score += 5.0
            elif guess.strip() == true_answer.strip():
                score += 3.5
            else:
                try:
                    ratio = float(guess) / float(true_answer)
                    if 0.9 <= ratio <= 1.1:
                        score += 2.0
                    elif 0.8 <= ratio <= 1.2:
                        score += 1.5
                    else:
                        score -= 2.5
                except Exception:
                    score -= 4.5
            scores.append(score)
        return scores

    match_numbers = re.compile(
        solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
        flags=re.MULTILINE | re.DOTALL,
    )
    print_state = {"times": 0}
    print_every = 1 if args.smoke_test else 5

    def check_numbers(prompts, completions, answer, **kwargs):
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [
            guess.group(1) if (guess := match_numbers.search(r)) is not None else None
            for r in responses
        ]
        if print_state["times"] % print_every == 0:
            print(
                "*" * 20
                + f"Question:\n{question}\nAnswer:\n{answer[0]}\nResponse:\n{responses[0]}\nExtracted:\n{extracted_responses[0]}"
            )
        print_state["times"] += 1

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(-2.5)
                continue
            try:
                true_answer_val = float(true_answer.strip())
                guess_val = float(guess.strip().replace(",", ""))
                scores.append(3.5 if guess_val == true_answer_val else -1.5)
            except Exception:
                scores.append(0)
        return scores

    def tokenize_prompt(batch):
        prompts = batch["prompt"]
        return {
            "tokens": [
                tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=True)
                for p in prompts
            ]
        }

    tokenized = dataset.map(tokenize_prompt, batched=True)
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
    maximum_length = int(np.quantile(tokenized["L"], 0.9))
    print("------------------", maximum_length, "------------------")
    dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])

    eval_sample_count = min(len(dataset), 16)# if args.smoke_test else 32) #Change
    eval_dataset = dataset.select(range(eval_sample_count)) if eval_sample_count else None

    max_prompt_length = maximum_length + 1
    max_completion_length = 1024#max_seq_length - max_prompt_length
    # if args.smoke_test:
    #     max_completion_length = min(max_completion_length, 64)

    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    num_generations = 4# if args.smoke_test else 4
    max_steps = 5 if args.smoke_test else 100

    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        save_steps=max_steps,
        report_to="none",
        output_dir=str(trainer_output_dir),
    )

    def evaluate_model(eval_data, lora_request=None):
        if eval_data is None or len(eval_data) == 0:
            return None
        samples = [eval_data[i] for i in range(len(eval_data))]
        prompts_messages = [sample["prompt"] for sample in samples]
        answers = [sample["answer"] for sample in samples]
        prompt_texts = [
            tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            for messages in prompts_messages
        ]
        eval_sampling_params = SamplingParams(
            temperature=1.0,
            top_k=50,
            max_tokens=max_completion_length,
        )
        outputs = model.fast_generate(
            prompt_texts,
            sampling_params=eval_sampling_params,
            lora_request=lora_request,
        )
        completions_text = [item.outputs[0].text for item in outputs]

        reward_lists = {
            "match_format_exactly": [],
            "match_format_approximately": [],
            "check_answer": [],
            "check_numbers": [],
            "total_reward": [],
        }

        for prompt, answer, completion in zip(prompts_messages, answers, completions_text):
            completion_struct = [[{"content": completion}]]
            prompt_struct = [prompt]
            answer_struct = [answer]
            mf_exact = match_format_exactly(completion_struct)[0]
            mf_approx = match_format_approximately(completion_struct)[0]
            ans_score = check_answer(prompt_struct, completion_struct, answer_struct)[0]
            num_score = check_numbers(prompt_struct, completion_struct, answer_struct)[0]
            total = mf_exact + mf_approx + ans_score + num_score
            reward_lists["match_format_exactly"].append(mf_exact)
            reward_lists["match_format_approximately"].append(mf_approx)
            reward_lists["check_answer"].append(ans_score)
            reward_lists["check_numbers"].append(num_score)
            reward_lists["total_reward"].append(total)

        summary = {
            "sample_count": len(samples),
            "averages": {k: float(np.mean(v)) for k, v in reward_lists.items()},
            "all_scores": reward_lists,
        }
        return summary

    initial_evaluation = evaluate_model(eval_dataset, lora_request=None)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    base_text = "What is the sqrt of 101?"
    sampling_params = SamplingParams(
        temperature=1.0,
        top_k=50,
        max_tokens=2048,
    )
    base_output = model.fast_generate(
        [base_text],
        sampling_params=sampling_params,
        lora_request=None,
    )[0].outputs[0].text
    print("Base model output:", base_output)

    if lora_output_path.exists():
        shutil.rmtree(lora_output_path)
    model.save_lora(str(lora_output_path))

    with safe_open(lora_output_path / "adapter_model.safetensors", framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            n_zeros = (tensor == 0).sum() / tensor.numel()
            assert n_zeros.item() != tensor.numel()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_text},
    ]
    chat_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    lora_output = model.fast_generate(
        [chat_text],
        sampling_params=SamplingParams(temperature=1.0, top_k=50, max_tokens=2048),
        lora_request=model.load_lora(str(lora_output_path)),
    )[0].outputs[0].text
    print("LoRA output:", lora_output)

    final_evaluation = evaluate_model(
        eval_dataset, lora_request=model.load_lora(str(lora_output_path))
    )

    generation_file = run_dir / "generation_outputs.txt"
    with open(generation_file, "w", encoding="utf-8") as handle:
        handle.write("Base model output:\n")
        handle.write(base_output.strip() + "\n\n")
        handle.write("LoRA output:\n")
        handle.write(lora_output.strip() + "\n")
    print(f"Saved run generations to {generation_file}")

    evaluation_file = run_dir / "evaluation_scores.json"
    percent_change = None
    if initial_evaluation and final_evaluation:
        initial_total = initial_evaluation["averages"]["total_reward"]
        final_total = final_evaluation["averages"]["total_reward"]
        if initial_total != 0:
            percent_change = ((final_total - initial_total) / abs(initial_total)) * 100.0
        else:
            percent_change = float("inf") if final_total != 0 else 0.0

    evaluation_summary = {
        "initial": initial_evaluation,
        "final": final_evaluation,
        "overall_percent_difference": percent_change,
    }
    with open(evaluation_file, "w", encoding="utf-8") as handle:
        json.dump(evaluation_summary, handle, indent=2)
    print(f"Saved evaluation results to {evaluation_file}")

    # Ensure distributed backends shut down cleanly to avoid CUDA allocator errors.
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    del trainer
    torch.cuda.empty_cache()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.flush()
    log_file.close()
    os._exit(0)


if __name__ == "__main__":
    main()
