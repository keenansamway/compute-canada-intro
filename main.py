import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

torch.manual_seed(42)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run language model inference')
    parser.add_argument('--model_id', type=str, required=False, help='Hugging Face model ID to use')
    parser.add_argument('--model_path', type=str, required=False, help='Path to Hugging Face model to use')
    args = parser.parse_args()

    model_id = args.model_id
    if model_id is None:
        model_id = args.model_path
        if model_id is None:
            print("Please provide a model ID or path to a local model to run.")
            print("Exiting...")
            exit(1)

    # Create the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
        # local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    print(f"Loaded {model_id} on {model.device}.")

    messages_batch = [
        # Example batch of messages for chat template
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain the theory of relativity in two sentences."},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the largest mammal on earth?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a haiku about the ocean."},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain the trolley problem in two sentences."},
        ],
    ]

    input_ids = tokenizer.apply_chat_template(
        messages_batch,
        return_tensors="pt",
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
    )

    outputs = model.generate(
        input_ids.to(model.device),
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
    )

    responses = tokenizer.batch_decode(
        outputs[:, input_ids.shape[1]:],  # Only take the new tokens generated
        skip_special_tokens=True,
    )


    # Save
    os.makedirs('results', exist_ok=True)
    with open('results/responses.txt', 'w') as f:
        for i, (messages, response) in enumerate(zip(messages_batch, responses)):
            question = messages[1]['content']
            f.write(f"Question {i+1}: {question}\n")
            f.write("-" * 70 + "\n")
            f.write(f"{response.strip()}\n\n\n")

    print(f"Responses saved to results/responses.txt.")


if __name__ == "__main__":
    main()
