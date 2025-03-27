#!/usr/bin/env python3
import sys
import argparse
from lcb_runner.runner.main import main as runner_main
from lcb_runner.lm_styles import LanguageModelStore, LanguageModel, LMStyle
from datetime import datetime
import os

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
        "--model_type",
        type=str,
        default="rl",
        help="one of rl, instruct, and base",
    )
    # Use parse_known_args to separate initializer args from the rest
    custom_args, remaining_args = parser.parse_known_args()
    
    model_name: str = custom_args.model
    model_type = custom_args.model_type
    safe_model_name = model_name.replace("/", "--")

    output_path = f"output/{safe_model_name}/Scenario.codegeneration_10_0.2_eval.json"
    if os.path.exists(output_path):
        return # we have already do this
    
    if model_type == "instruct":
        model = LanguageModel(
            model_name,
            safe_model_name,
            LMStyle.CodeQwenInstruct,
            datetime(2025, 3, 25),
            link=f"https://huggingface.co/{model_name}",
        )
    elif model_type == "base":
        model = LanguageModel(
            model_name,
            safe_model_name,
            LMStyle.GenericBase,
            datetime(2025, 3, 25),
            link=f"https://huggingface.co/{model_name}",
        )
    elif model_type == "rl":
        model = LanguageModel(
            model_name,
            safe_model_name,
            LMStyle.AceCoderV2RL,
            datetime(2025, 3, 25),
            link=f"https://huggingface.co/{model_name}",
        )
    else:
        raise Exception(f"unknown model type: {model_type}, currently only support rl, instruct and base")
        
    LanguageModelStore[model.model_name] = model

    # Update sys.argv for the runner_main, so that its argparse sees only its expected arguments.
    sys.argv = [sys.argv[0], "--model", model_name] + remaining_args

    # Call the original main function
    runner_main()

if __name__ == "__main__":
    main()
