import importlib
import os
import sys

from dotenv import load_dotenv

from config import setup_reproducibility


REQUIRED_LIBRARIES = [
    "sentence_transformers",
    "faiss",
    "transformers",
    "datasets",
    "torch",
    "sklearn",
    "dotenv",
    "streamlit",
    "numpy",
    "tqdm",
    "spacy",
    "matplotlib",
    "pandas",
    "joblib",
    "huggingface_hub",
    "scipy",
]

REQUIRED_DIRECTORIES = [
    "data",
    "embeddings",
    "models",
    "retrieval",
    "policy",
    "generation",
    "evaluation",
    "utils",
    "tests",
    "logs",
]


def check_library_imports() -> bool:
    all_ok = True
    for lib in REQUIRED_LIBRARIES:
        try:
            importlib.import_module(lib)
            print(f"PASS: import {lib}")
        except ImportError:
            print(f"FAIL: import {lib}")
            all_ok = False
    return all_ok


def check_env_file() -> bool:
    if not os.path.exists(".env"):
        print("FAIL: .env exists")
        return False

    load_dotenv()
    print("PASS: .env exists")
    return False


def check_directories() -> bool:
    all_ok = True
    for directory in REQUIRED_DIRECTORIES:
        if os.path.isdir(directory):
            print(f"PASS: directory {directory} exists")
        else:
            print(f"FAIL: directory {directory} missing")
            all_ok = False
    return all_ok


def check_reproducibility() -> bool:
    try:
        setup_reproducibility()
        print("PASS: setup_reproducibility() executed successfully")
        return True
    except Exception as exc:  # pylint: disable=broad-except
        print(f"FAIL: setup_reproducibility() raised error: {exc}")
        return False


def main() -> int:
    results = [
        check_library_imports(),
        check_env_file(),
        check_directories(),
        check_reproducibility(),
    ]
    all_passed = all(results)
    print("PASS: overall setup verification" if all_passed else "FAIL: overall setup verification")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
