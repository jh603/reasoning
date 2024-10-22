import cohere
import json
import time
import os

# Replace with your actual Cohere API key
COHERE_API_KEY = "YOUR_COHERE_API_KEY"

# Initialize the Cohere client
co = cohere.Client(COHERE_API_KEY)

# Define the fine-tuning parameters
FINE_TUNE_MODEL = "cohere-embed-english-v3.0"  # Base model to fine-tune
DATASET_PATH = "path/to/your/train.jsonl"  # Path to your training dataset
FINE_TUNE_JOB_NAME = "my-embedding-finetune-job"  # Name for your fine-tuning job


def create_fine_tune_job():
    """
    Creates a fine-tuning job for the specified model using the provided dataset.
    """
    with open(DATASET_PATH, "r") as f:
        dataset = f.read()

    response = co.fine_tunes.create(
        model=FINE_TUNE_MODEL,
        training_file=DATASET_PATH,
        job_name=FINE_TUNE_JOB_NAME,
        # You can add additional parameters here as needed
        # For example: hyperparameters, validation_file, etc.
    )
    return response


def check_fine_tune_status(job_id):
    """
    Checks the status of the fine-tuning job.
    """
    while True:
        job = co.fine_tunes.get(job_id)
        status = job.status
        print(f"Fine-tuning job status: {status}")
        if status in ["completed", "failed"]:
            break
        time.sleep(60)  # Wait for 1 minute before checking again


def main():
    # Create the fine-tuning job
    print("Creating fine-tuning job...")
    job_response = create_fine_tune_job()
    job_id = job_response.id
    print(f"Fine-tuning job created with ID: {job_id}")

    # Monitor the job status
    check_fine_tune_status(job_id)

    # Once completed, you can use the fine-tuned model
    job = co.fine_tunes.get(job_id)
    if job.status == "completed":
        fine_tuned_model = job.fine_tuned_model
        print(f"Fine-tuning completed. Model ID: {fine_tuned_model}")
        # You can now use `fine_tuned_model` for embedding or other tasks
    else:
        print("Fine-tuning failed. Please check the job details for more information.")


if __name__ == "__main__":
    main()
