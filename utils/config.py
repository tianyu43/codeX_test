import argparse
import json
from typing import Any, Dict


def load_json_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8-sig") as file_obj:
        config = json.load(file_obj)
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a JSON object at the top level.")
    return config


def parse_args_with_config(build_parser) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, remaining_argv = config_parser.parse_known_args()

    parser = build_parser()
    if config_args.config:
        parser.set_defaults(**load_json_config(config_args.config))

    args = parser.parse_args(remaining_argv)
    setattr(args, "config", config_args.config)
    return args
