#!/usr/bin/env python3

import csv
import torch
from torch.utils.data import Dataset 
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)
import numpy as np


class EnigmaDataset(Dataset):
    def __init__(self, texts, scrambles, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.input_texts = scrambles
        self.target_texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_encoding = self.tokenizer(
            self.input_texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            self.target_texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }
    

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    labels[labels == -100] = 0 
  
    correct = (predictions == labels).sum()
    total = (labels != 0).sum() 
    
    return {"accuracy": correct.item() / total.item() if total.item() > 0 else 0.0}


def main():
    text = []
    scrambles = []
    firstline = 1
    line_count = 0
    
    with open('../Data/enigma_processed.csv', newline='', encoding='utf-8') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for line in lines:
            if firstline:
                firstline = 0
                continue
            text.append(line[0].lower()) 
            scrambles.append(line[1].lower())
            line_count += 1

    split_point = int(0.8 * line_count)

    training_text = text[:split_point]
    training_scrambles = scrambles[:split_point]
    validation_text = text[split_point:line_count]
    validation_scrambles = scrambles[split_point:line_count]
    
    MODEL_NAME = "google/byt5-small"
    OUTPUT_DIR = "../byt5_enigma_finetuned"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


    train_dataset = EnigmaDataset(training_text, training_scrambles, tokenizer)
    eval_dataset = EnigmaDataset(validation_text, validation_scrambles, tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1, 
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        save_strategy="epoch", 
        fp16=torch.cuda.is_available(), 
        predict_with_generate=True, 
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding="longest",
        label_pad_token_id=tokenizer.pad_token_id
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting Fine-tuning...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Fine-tuning complete. Model saved to {OUTPUT_DIR}")

    results = trainer.evaluate()
    print(f"Final Evaluation Results: {results}")

if __name__ == '__main__':
    main()