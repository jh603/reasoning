import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


# Manually load model from checkpoint and verify it has the correct layers
def load_model_from_checkpoint(model_name, checkpoint_path):
    # Load tokenizer as usual
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the pre-trained model architecture (without weights)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Load the fine-tuned weights from the PyTorch checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Check if the QA output layers are present in the checkpoint
    if "qa_outputs.weight" in checkpoint and "qa_outputs.bias" in checkpoint:
        print("QA output layers found in the checkpoint.")
    else:
        print(
            "WARNING: QA output layers not found in the checkpoint. These layers will be randomly initialized."
        )

    # Load the state dict with strict=False to allow partial load
    model.load_state_dict(checkpoint, strict=False)

    # Return the tokenizer and model
    return tokenizer, model


# Function to perform inference
def predict_answer(question, context, tokenizer, model, device):
    # Tokenize the input
    inputs = tokenizer(question, context, return_tensors="pt")

    # Move input tensors to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the start and end logits
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Find the start and end token positions
    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits)

    # Extract the answer tokens
    answer_tokens = inputs["input_ids"][0][start_idx : end_idx + 1]

    # Decode the tokens into text
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer


# Main function
def main():
    # Base model name
    model_name = (
        "microsoft/deberta-v3-large"  # This is the base model you used for fine-tuning
    )

    # Path to the checkpoint file
    checkpoint_path = "/home/cpp/jerryhuang/beam_retriever/output/10-10-2024/hotpotqa_reader_deberta_large-seed42-bsz4-fp16True-lr1e-05-decay0.0-warm0.1-valbsz32/checkpoint_best.pt"

    # Load the model and tokenizer with checkpoint checks
    tokenizer, model = load_model_from_checkpoint(model_name, checkpoint_path)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Take input from user
    question = input("Enter your question: ")
    context = input("Enter the context: ")

    # Get the answer
    answer = predict_answer(question, context, tokenizer, model, device)

    # Output the answer
    print(f"Predicted Answer: {answer}")


if __name__ == "__main__":
    main()
