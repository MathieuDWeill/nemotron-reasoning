#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


from nemotron_reasoning.data import load_train_csv, make_hf_dataset
from nemotron_reasoning.utils import ensure_dir, load_yaml, patch_nemotron_triton, seed_everything


def resolve_model_path(cfg):
    if cfg.get("model_path"):
        return cfg["model_path"]

    if cfg.get("model_handle"):
        import kagglehub
        return kagglehub.model_download(cfg["model_handle"])

    raise ValueError("Config must define either model_path or model_handle.")


def _import_training_deps():
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/sft_default.yaml')
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get('seed', 42)))
    patch_nemotron_triton()

    model_path = resolve_model_path(cfg)
    output_dir = ensure_dir(cfg['output_dir'])

    df = load_train_csv(cfg['train_csv'])
    if cfg.get('subsample_size'):
        n = min(int(cfg['subsample_size']), len(df))
        df = df.sample(n=n, random_state=int(cfg.get('seed', 42)))
    print('Training rows:', len(df))
    print(df['family'].value_counts().to_string())

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = make_hf_dataset(
        df,
        tokenizer,
        use_chat_template=bool(cfg.get('formatting', {}).get('use_chat_template', True)),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    lcfg = cfg['lora']
    lora_config = LoraConfig(
        r=int(lcfg.get('r', 32)),
        lora_alpha=int(lcfg.get('alpha', 32)),
        target_modules=lcfg.get('target_modules', 'all-linear'),
        lora_dropout=float(lcfg.get('dropout', 0.05)),
        bias=lcfg.get('bias', 'none'),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(cfg.get('per_device_train_batch_size', 1)),
        gradient_accumulation_steps=int(cfg.get('gradient_accumulation_steps', 8)),
        num_train_epochs=float(cfg.get('num_train_epochs', 1)),
        learning_rate=float(cfg.get('learning_rate', 5e-5)),
        logging_steps=int(cfg.get('logging_steps', 5)),
        bf16=True,
        max_grad_norm=1.0,
        optim='adamw_torch',
        lr_scheduler_type=cfg.get('lr_scheduler_type', 'cosine'),
        warmup_ratio=float(cfg.get('warmup_ratio', 0.1)),
        save_strategy=cfg.get('save_strategy', 'no'),
        report_to='none',
        dataset_text_field='text',
        max_length=int(cfg.get('max_seq_len', 2048)),
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
    )

    trainer = SFTTrainer(model=model, args=sft_args, train_dataset=dataset)
    trainer.train()
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir / 'tokenizer'))
    print('Saved adapter to', output_dir)

if __name__ == '__main__':
    main()
