#!/usr/bin/env python3
import sys
import argparse
from lcb_runner.runner.main import main as runner_main
from lcb_runner.lm_styles import LanguageModelStore, LanguageModel, LMStyle
from datetime import datetime


def main():
    # Create an argument parser for the initializer's custom arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="the model",
    )
    parser.add_argument(
        "--base",
        type=bool,
        default=True,
        help="set to True if its qwen base model, else we assume instruct model",
    )
    # Use parse_known_args to separate initializer args from the rest
    custom_args, remaining_args = parser.parse_known_args()
    
    model_name: str = custom_args.model
    base_model = custom_args.base
    
    if not base_model:
        model = LanguageModel(
            model_name,
            model_name.replace("/", "--"),
            LMStyle.CodeQwenInstruct,
            datetime(2025, 3, 25),
            link=f"https://huggingface.co/{model_name}",
        )
    else:
        model = LanguageModel(
            model_name,
            model_name.replace("/", "--"),
            LMStyle.GenericBase,
            datetime(2025, 3, 25),
            link=f"https://huggingface.co/{model_name}",
        )
        
    LanguageModelStore[model.model_name] = model

    # Update sys.argv for the runner_main, so that its argparse sees only its expected arguments.
    sys.argv = [sys.argv[0], "--model", model_name] + remaining_args

    # Call the original main function
    runner_main()

if __name__ == "__main__":
    main()
