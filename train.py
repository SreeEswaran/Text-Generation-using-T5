import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def train():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Prepare dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="data/input.txt",
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir="./model/t5",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model("./model/t5")

if __name__ == "__main__":
    train()
