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

import duckdb
import evaluation as evaluation_module
from evaluation import evaluate as sql_evaluate


SQL_DUCKDB_CONN = None

SYSTEM_PROMPT = (
    "You are an expert DuckDB SQL assistant.\n"
    "Take the user request, problem background, and SQL context and return a DuckDB-compatible SQL query.\n" 
    "Use any DuckDB clauses, functions, aggregations, or joins needed to make the best query possible\n"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT + GRPO fine-tuning for Qwen3-4B.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where GRPO checkpoints and logs are stored.",
    )
    parser.add_argument(
        "--base-model",
        default="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
        help="Huggingface format name of the base model to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Seed to use for the model tuning",
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
        "--training-data-csv",
        default="training_data.csv",
        help="Path to the GRPO training CSV (your attack detection prompts).",
    )
    parser.add_argument(
        "--duckdb-db-path",
        default="../data/data.duckdb",
        help="Path to the DuckDB database file used by evaluation.py.",
    )
    parser.add_argument(
        "--sft-max-steps",
        type=int,
        default=25000,
        help="Max steps for the sft stage",
    )
    parser.add_argument(
        "--sft-save-steps",
        type=int,
        default=500,
        help="How often (in steps) to save SFT checkpoints. Defaults to 500.",
    )
    parser.add_argument(
        "--grpo-save-steps",
        type=int,
        default=100,
        help="How often (in steps) to save GRPO checkpoints. Defaults to 100.",
    )
    parser.add_argument(
        "--grpo-max-steps",
        type=int,
        default=500,
        help="Max steps for the grpo stage",
    )
    parser.add_argument(
        "--resume-grpo-from-checkpoint",
        default=None,
        help=(
            "Path to a previous GRPO Trainer checkpoint directory to resume from "
            "(e.g. .../trainer_outputs/checkpoint-50)."
        ),
    )
    parser.add_argument(
        "--resume-sft-from-checkpoint",
        default=None,
        help=(
            "Path to a previous SFT checkpoint directory to resume SFT from "
            "(e.g. .../sft_outputs/checkpoint-500)."
        ),
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
    run_dir: Path, args: argparse.Namespace, extra: dict | None = None) -> None:
    metadata = {
        "command": " ".join(sys.argv),
        "args": vars(args),
        "run_directory": str(run_dir),
    }
    if extra:
        metadata["extra_paths"] = extra
    with open(run_dir / "run_args.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)


def format_sft_example(row: pd.Series) -> list[dict[str, str]]:
    """Turn a motherduckdb/duckdb-text2sql-25k row into chat messages."""
    prompt = row["prompt"]
    schema = row.get("schema", "") if "schema" in row else ""
    if isinstance(schema, float) and pd.isna(schema):
        schema = ""

    if schema:
        user_content = (
            "Write a DuckDB SQL query that solves the request\n"
            "SQL Context:\n"
            f"{schema}\n\n"
            "Request:\n"
            f"{prompt}"
        )
    else:
        user_content = (
             "Write a DuckDB SQL query that solves the request\n\n"
            f"Request:\n{prompt}"
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": row["query"].strip()},
    ]


def build_sft_dataset(tokenizer, max_seq_length: int, smoke_test: bool) -> Dataset:
    """Load and format the SFT dataset from motherduckdb/duckdb-text2sql-25k."""
    keep_categories = [
        "guides/sql_features/asof_join",
        "guides/sql_features/full_text_search",
        "sql/aggregates",
        "sql/case_sensitivity",
        "sql/configuration",
        "sql/constraints",
        "sql/data_types/bitstring",
        "sql/data_types/blob",
        "sql/data_types/boolean",
        "sql/data_types/date",
        "sql/data_types/enum",
        "sql/data_types/interval",
        "sql/data_types/list",
        "sql/data_types/map",
        "sql/data_types/nulls",
        "sql/data_types/numeric",
        "sql/data_types/overview",
        "sql/data_types/struct",
        "sql/data_types/text",
        "sql/data_types/time",
        "sql/data_types/timestamp",
        "sql/data_types/timezones",
        "sql/data_types/union",
        "sql/duckdb_table_functions",
        "sql/expressions/case",
        "sql/expressions/cast",
        "sql/expressions/collations",
        "sql/expressions/comparison_operators",
        "sql/expressions/in",
        "sql/expressions/logical_operators",
        "sql/expressions/overview",
        "sql/expressions/star",
        "sql/expressions/subqueries",
        "sql/functions/bitstring",
        "sql/functions/blob",
        "sql/functions/char",
        "sql/functions/date",
        "sql/functions/dateformat",
        "sql/functions/datepart",
        "sql/functions/enum",
        "sql/functions/interval",
        "sql/functions/nested",
        "sql/functions/numeric",
        "sql/functions/overview",
        "sql/functions/patternmatching",
        "sql/functions/time",
        "sql/functions/timestamp",
        "sql/functions/timestamptz",
        "sql/functions/utility",
        "sql/indexes",
        "sql/information_schema",
        "sql/introduction",
        "sql/query_syntax/filter",
        "sql/query_syntax/from",
        "sql/query_syntax/groupby",
        "sql/query_syntax/grouping_sets",
        "sql/query_syntax/having",
        "sql/query_syntax/limit",
        "sql/query_syntax/orderby",
        "sql/query_syntax/qualify",
        "sql/query_syntax/sample",
        "sql/query_syntax/select",
        "sql/query_syntax/setops",
        "sql/query_syntax/unnest",
        "sql/query_syntax/values",
        "sql/query_syntax/where",
        "sql/query_syntax/window",
        "sql/query_syntax/with",
        "sql/samples",
        "sql/statements/create_macro",
        "sql/statements/create_schema",
        "sql/statements/create_sequence",
        "sql/statements/pivot",
        "sql/statements/select",
        "sql/statements/unpivot",
        "sql/window_functions"
    ]
    ds = load_dataset("motherduckdb/duckdb-text2sql-25k", split="train")
    df_temp = ds.to_pandas()
    keep_filter = df_temp.category.isin(keep_categories)
    df = df_temp[keep_filter][["prompt", "query", "schema"]]

    if smoke_test:
        df = df.head(512).copy()

    df["Messages"] = df.apply(format_sft_example, axis=1)
    df["N"] = df["Messages"].apply(
        lambda x: len(tokenizer.apply_chat_template(x))
    )
    # Keep samples that fit into context
    filtered_df = df.loc[df["N"] <= max_seq_length].copy()
    if filtered_df.empty:
        filtered_df = df.head(1).copy()

    filtered_df["text"] = tokenizer.apply_chat_template(
        filtered_df["Messages"].values.tolist(), tokenize=False
    )
    sft_dataset = Dataset.from_pandas(filtered_df, preserve_index=False)
    return sft_dataset


def sql_reward(
    prompts,
    completions,
    column,
    value,
    target_score,
    total_rows,
    **kwargs,):
    """Reward function that calls evaluation.evaluate on each generated SQL query.

    Returns a list of F2 scores (with penalties from evaluation.py).
    """
    scores: list[float] = []
    global SQL_DUCKDB_CONN

    # Ensure the evaluation module sees the same DuckDB connection
    evaluation_module.con = SQL_DUCKDB_CONN

    for completion, col, val, tgt, total in zip(
        completions, column, value, target_score, total_rows
    ):
        # completions are structured as [[{"content": "..."}], ...]
        response = completion[0]["content"]
        query = response.strip()
        try:
            result = sql_evaluate(SQL_DUCKDB_CONN, query, col, val, tgt, total)
            scores.append(float(result.get("f2", 0.0)))
        except Exception:
            # If anything goes wrong, fall back to a strong negative reward.
            scores.append(-20.0)
    return scores


def evaluate_model(
    eval_data: Dataset,
    tokenizer,
    max_completion_length: int,
    model,
    duckdb_conn,
    lora_request=None,):
    """Run a small evaluation loop and summarize average F2 over eval_data."""
    if eval_data is None or len(eval_data) == 0:
        return None

    evaluation_module.con = duckdb_conn

    samples = [eval_data[i] for i in range(len(eval_data))]
    prompts_messages = [sample["prompt"] for sample in samples]
    prompt_texts = [
        tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        for messages in prompts_messages
    ]

    eval_sampling_params = SamplingParams(
        temperature=0.7,
        top_k=50,
        max_tokens=max_completion_length,
    )
    outputs = model.fast_generate(
        prompt_texts,
        sampling_params=eval_sampling_params,
        lora_request=lora_request,
    )
    completions_text = [item.outputs[0].text for item in outputs]

    f2_scores: list[float] = []
    detailed_scores = []

    for sample, completion in zip(samples, completions_text):
        query = completion.strip()
        if query.startswith("```"):
            query = re.sub(r"^```(?:sql)?", "", query, flags=re.IGNORECASE).strip()
            if query.endswith("```"):
                query = query[:-3].strip()

        col = sample["column"]
        val = sample["value"]
        tgt = sample["target_score"]
        total = sample["total_rows"]

        try:
            result = sql_evaluate(duckdb_conn, query, col, val, tgt, total)
        except Exception:
            result = {
                "output": None,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": -20.0,
                "f2": -20.0,
            }
        f2_scores.append(float(result.get("f2", 0.0)))
        detailed_scores.append(
            {
                "prompt": sample["prompt"],
                "generated_sql": query,
                "metrics": result,
            }
        )

    summary = {
        "sample_count": len(samples),
        "average_f2": float(np.mean(f2_scores)) if f2_scores else 0.0,
        "all_scores": detailed_scores,
    }
    return summary


def build_grpo_dataset(
    tokenizer,
    training_csv_path: str,
    smoke_test: bool,):
    """Build the GRPO dataset from your training_data.csv file.

    Expects columns:
        - SQL Context
        - Question
        - Answer Field
        - Answer
        - Total Rows
        - Attack Type
        - Question Context
    """
    df = pd.read_csv(training_csv_path)
    required_cols = [
        "SQL Context",
        "Question",
        "Answer Field",
        "Answer",
        "Total Rows",
        "Attack Type",
        "Question Context",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"training_data.csv is missing required columns: {missing}")

    if smoke_test:
        df = df.head(32).copy()

    prompts = []
    columns = []
    values = []
    target_scores = []
    total_rows = []

    for _, row in df.iterrows():
        sql_context = str(row["SQL Context"]).strip()
        question = str(row["Question"]).strip()
        answer_field = str(row["Answer Field"]).strip()
        attack_type = str(row["Attack Type"]).strip()
        question_context = str(row["Question Context"]).strip()

        user_content = (
            f"{question_context}\n"
            "Request:\n"
            f"{question}\n\n"
            "SQL Context:\n\n"
            f"{sql_context}\n\n"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        prompts.append(messages)
        columns.append(answer_field)
        values.append(attack_type)
        target_scores.append(int(row["Answer"]))
        total_rows.append(int(row["Total Rows"]))

    dataset = Dataset.from_dict(
        {
            "prompt": prompts,
            "column": columns,
            "value": values,
            "target_score": target_scores,
            "total_rows": total_rows,
        }
    )

    def tokenize_prompt(batch):
        msgs = batch["prompt"]
        return {
            "tokens": [
                tokenizer.apply_chat_template(
                    p, add_generation_prompt=True, tokenize=True
                )
                for p in msgs
            ]
        }

    tokenized = dataset.map(tokenize_prompt, batched=True)
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
    lengths = np.array(tokenized["L"])
    maximum_length = int(np.quantile(lengths, 0.9)) if len(lengths) else 128
    print("------------------ Prompt length p90:", maximum_length, "------------------")
    keep_indices = np.where(lengths <= maximum_length)[0]
    dataset = dataset.select(keep_indices.tolist())

    eval_sample_count = min(len(dataset), 8 if smoke_test else 32)
    eval_dataset = dataset.select(range(eval_sample_count)) if eval_sample_count else None

    return dataset, eval_dataset, maximum_length

def main() -> None:
    global SQL_DUCKDB_CONN

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = prepare_run_directory(Path(args.output_dir))
    log_file, original_stdout, original_stderr = init_run_logging(run_dir)
    
    sft_output_dir = run_dir / "sft_outputs"
    sft_output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer_output_dir = run_dir / "trainer_outputs"
    lora_output_path = run_dir / Path(args.lora_output).name
    trainer_output_dir.mkdir(parents=True, exist_ok=True)
    record_run_arguments(
        run_dir,
        args,
        extra={
            "sft_output_dir": str(sft_output_dir),
            "trainer_output_dir": str(trainer_output_dir),
            "lora_output_dir": str(lora_output_path),
        },
    )
    print(f"Run artifacts will be stored in: {run_dir}")

    SQL_DUCKDB_CONN = duckdb.connect(args.duckdb_db_path)
    evaluation_module.con = SQL_DUCKDB_CONN

    max_seq_length = 4096
    lora_rank = 8
    load_in_4bit = True
    gpu_memory_utilization = 0.75 
    conservativeness = 0.5

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        use_flash_attention_2=True,
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
        random_state=args.seed,
    )

    SYSTEM_PROMPT = (
        "You are an expert DuckDB SQL assistant.\n"
        "Take the user request, problem background, and SQL context and return a DuckDB-compatible SQL query.\n" 
        "Use any DuckDB clauses, functions, aggregations, or joins needed to make the best query possible\n"
    )

    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + eos_token }}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        "{{ '" + SYSTEM_PROMPT.replace("'", "\\'") + "' + eos_token }}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ message['content'] + eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '' }}{% endif %}"
    )
    tokenizer.chat_template = chat_template

    # -----------------------
    # 1) SFT on text-to-SQL
    # -----------------------
    print("Building SFT dataset from motherduckdb/duckdb-text2sql-25k ...")
    sft_dataset = build_sft_dataset(tokenizer, max_seq_length, args.smoke_test)

    sft_epochs = 1 if args.smoke_test else 2
    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=sft_dataset,
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
            seed=args.seed,
            report_to="none",
            output_dir=str(sft_output_dir),
            max_steps=100 if args.smoke_test else args.sft_max_steps,
            save_steps=args.sft_save_steps,
            gradient_checkpointing=True,
        ),
    )
    if args.resume_sft_from_checkpoint:
        print(
            f"Resuming SFT training from checkpoint: "
            f"{args.resume_sft_from_checkpoint}"
        )
        sft_trainer.train(resume_from_checkpoint=args.resume_sft_from_checkpoint)
    else:
        sft_trainer.train()

    # Quick sanity generation after SFT
    example_messages = sft_dataset[0]["Messages"][:2]
    demo_text = tokenizer.apply_chat_template(
        example_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    _ = model.generate(
        **tokenizer(demo_text, return_tensors="pt").to(device),
        temperature=0.7,
        max_new_tokens=1024,
        streamer=TextStreamer(tokenizer, skip_prompt=False),
    )

    del sft_dataset
    torch.cuda.empty_cache()
    gc.collect()

    # -----------------------
    # 2) GRPO with evaluation.py on training_data.csv
    # -----------------------
    print("Building GRPO dataset from training_data.csv ...")
    grpo_dataset, eval_dataset, maximum_length = build_grpo_dataset(
        tokenizer,
        args.training_data_csv,
        args.smoke_test,
    )

    max_prompt_length = maximum_length + 1
    max_completion_length = 1024

    vllm_sampling_params = SamplingParams(
        min_p=0,
        top_p=0.8,
        top_k=20,
        temperature=0.7,
        seed=args.seed,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    num_generations = 2 if args.smoke_test else 6
    max_steps = 5 if args.smoke_test else args.grpo_max_steps
    save_steps = args.grpo_save_steps
    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=0.7,
        learning_rate=5e-6,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=20,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        save_steps=save_steps,
        report_to="none",
        output_dir=str(trainer_output_dir),
    )

    if args.resume_grpo_from_checkpoint:
        print(
            f"Resume-from-checkpoint specified "
            f"({args.resume_grpo_from_checkpoint}); skipping initial pre-GRPO evaluation."
        )
        initial_evaluation = None
    else:
        print("Running initial evaluation before GRPO ...")
        initial_evaluation = evaluate_model(
            eval_dataset,
            tokenizer,
            max_completion_length,
            model,
            duckdb_conn=SQL_DUCKDB_CONN,
            lora_request=None,
        )



    grpo_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[sql_reward],
        args=training_args,
        train_dataset=grpo_dataset,
    )
    if args.resume_grpo_from_checkpoint:
        print(f"Resuming GRPO training from checkpoint: {args.resume_grpo_from_checkpoint}")
        grpo_trainer.train(resume_from_checkpoint=args.resume_grpo_from_checkpoint)
    else:
        grpo_trainer.train()


    # -----------------------
    # 3) Save LoRA and compare generations
    # -----------------------
    demo_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "You have a DuckDB table named logs with columns "
                "(timestamp, src_ip, dst_ip, dst_port, bytes). "
                "Write a query that returns all rows where dst_port = 22 "
                "and bytes > 0."
            ),
        },
    ]
    demo_chat = tokenizer.apply_chat_template(
        demo_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    sampling_params = SamplingParams(
        temperature=0.9,
        top_k=50,
        max_tokens=256,
    )
    base_output = model.fast_generate(
        [demo_chat],
        sampling_params=sampling_params,
        lora_request=None,
    )[0].outputs[0].text
    print("Base (in-run) model output:", base_output)

    if lora_output_path.exists():
        shutil.rmtree(lora_output_path)
    model.save_lora(str(lora_output_path))

    with safe_open(lora_output_path / "adapter_model.safetensors", framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            n_zeros = (tensor == 0).sum() / tensor.numel()
            assert n_zeros.item() != tensor.numel()

    lora_output = model.fast_generate(
        [demo_chat],
        sampling_params=SamplingParams(temperature=0.9, top_k=50, max_tokens=256),
        lora_request=model.load_lora(str(lora_output_path)),
    )[0].outputs[0].text
    print("LoRA output:", lora_output)

    print("Running final evaluation with LoRA loaded ...")
    final_evaluation = evaluate_model(
        eval_dataset,
        tokenizer,
        max_completion_length,
        model,
        duckdb_conn=SQL_DUCKDB_CONN,
        lora_request=model.load_lora(str(lora_output_path)),
    )

    generation_file = run_dir / "generation_outputs.txt"
    with open(generation_file, "w", encoding="utf-8") as handle:
        handle.write("Base model output (no LoRA loaded):\n")
        handle.write(base_output.strip() + "\n\n")
        handle.write("LoRA output:\n")
        handle.write(lora_output.strip() + "\n")
    print(f"Saved run generations to {generation_file}")

    evaluation_file = run_dir / "evaluation_scores.json"
    percent_change = None
    if initial_evaluation and final_evaluation:
        initial_val = initial_evaluation.get("average_f2", 0.0)
        final_val = final_evaluation.get("average_f2", 0.0)
        if initial_val != 0:
            percent_change = ((final_val - initial_val) / abs(initial_val)) * 100.0
        else:
            percent_change = float("inf") if final_val != 0 else 0.0

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

    del grpo_trainer
    torch.cuda.empty_cache()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.flush()
    log_file.close()
    SQL_DUCKDB_CONN.close()
    os._exit(0)


if __name__ == "__main__":
    main()
