from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer

def load_dataset(train_path):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128,
    )

if __name__ == '__main__':
    # Initialize the GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Load the training dataset
    train_dataset = load_dataset("train.txt")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    # Initialize Trainer
    training_args = TrainingArguments(
        output_dir="./output",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("./output")
