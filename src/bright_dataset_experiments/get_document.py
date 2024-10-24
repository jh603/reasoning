"""
get_document.py

A script to retrieve and display the content of a specific document from a dataset subset and split.

Usage:
    python get_document.py --subset <subset_name> --split <split_name> --document_id <document_id>

Example:
    python get_document.py --subset llama3-70b_reason --split stackoverflow --document_id 12345
"""

import argparse
import re
import sys
from datasets import load_dataset


def sanitize_input(s: str) -> str:
    """
    Sanitize input strings to prevent injection or filesystem issues.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Retrieve Document Content from Dataset"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="documents",
        required=False,
        help="Dataset subset to use (e.g., llama3-70b_reason)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="stackoverflow",
        required=False,
        help="Dataset split to use (e.g., stackoverflow)",
    )
    parser.add_argument(
        "--document_id",
        type=str,
        default="cloudformation_commands/cloudformation_commands_58_1.txt",
        required=False,
        help="ID of the document to retrieve",
    )
    return parser.parse_args()


def get_document_content(dataset, document_id):
    """
    Retrieve the content of the document with the specified ID.

    Args:
        dataset: The loaded dataset split.
        document_id (str): The ID of the document to retrieve.

    Returns:
        str: Content of the document if found, else None.
    """
    try:
        # Use filter to find the document
        filtered = dataset.filter(lambda x: str(x["id"]) == str(document_id))
        if len(filtered) > 0:
            return filtered[0]["content"]
        else:
            return None
    except KeyError:
        print("The dataset does not contain 'id' or 'content' fields.")
        return None
    except Exception as e:
        print(f"Error retrieving document: {e}")
        return None


def main():
    args = parse_args()

    subset = sanitize_input(args.subset)
    split = sanitize_input(args.split)
    document_id = args.document_id

    print(f"Loading dataset subset '{subset}' and split '{split}'...")

    try:
        # Load the dataset. Adjust "xlangai/BRIGHT" if your dataset path is different.
        dataset = load_dataset("xlangai/BRIGHT", subset, split=split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    print(f"Searching for document ID: {document_id}...")

    content = get_document_content(dataset, document_id)

    if content:
        print(f"\nContent of Document ID {document_id}:\n")
        print(content)
    else:
        print(
            f"Document with ID '{document_id}' not found in subset '{subset}' and split '{split}'."
        )


if __name__ == "__main__":
    main()
