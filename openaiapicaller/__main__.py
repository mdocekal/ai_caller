import base64
import csv
import json
import logging
import os.path
import sys
from argparse import ArgumentParser
from contextlib import nullcontext
from io import StringIO, BytesIO
from json import JSONDecodeError
from math import ceil
from pathlib import Path
from typing import Sequence, Optional, TextIO

import inquirer
import numpy as np
import requests
import tiktoken
from PIL import Image
from classconfig import Config, ConfigurableFactory, ConfigurableMixin, ConfigurableSubclassFactory
from classconfig.classes import subclasses
from tqdm import tqdm

from openaiapicaller.api import API
from openaiapicaller.conversion import Convertor
from openaiapicaller.loader import Loader
from openaiapicaller.sample_assembler import APISampleAssembler, TemplateBasedAssembler
from openaiapicaller.template import Template
from openaiapicaller.utils import read_potentially_malformed_json_result

SCRIPT_DIR = Path(__file__).parent


class CreateBatchWorkflow(ConfigurableMixin):
    """
    Workflow for creating batch file for OpenAI API.
    """

    convertor: Convertor = ConfigurableSubclassFactory(Convertor, "Convertor to batch file.")

    def __call__(self, proc_path: Optional[str] = None):
        """
        Creates batch file for OpenAI API. Results are printed to stdout.

        :param proc_path: Path to data.
        """
        for req in self.convertor.convert(proc_path):
            print(req)


def create_batch_file(args):
    """
    Creates batch file for OpenAI API. Results are printed to stdout.

    :param args: Parsed arguments.
    """

    config_path = Path(args.config) if args.config is not None else SCRIPT_DIR / "configs" / "create_batch_file.yaml"
    config = Config(CreateBatchWorkflow).load(config_path)
    workflow = ConfigurableFactory(CreateBatchWorkflow).create(config)
    workflow(proc_path=args.path)


def load_requests_ids(file: str) -> set[str]:
    """
    Loads request ids from the file.

    :param file: Path to the file.
    :return: Set of request ids.
    """
    with open(file, mode='r') as f:
        return {json.loads(l)["custom_id"] for l in f}


class APISelector(ConfigurableMixin):
    api: API = ConfigurableSubclassFactory(API, "API type")


def batch_request(args):
    """
    Sends requests to OpenAI API.

    :param args: Parsed arguments.
    """
    config_path = Path(args.config) if args.config is not None else SCRIPT_DIR / "configs" / "api.yaml"
    config = Config(APISelector).load(config_path)
    api = ConfigurableFactory(APISelector).create(config).api

    if args.line is None:
        if args.results is None:
            raise ValueError("Results file must be specified when sending batch requests.")

        p = Path(args.file)
        with open(args.results, mode=('a' if args.cont else 'w')) if args.results is not None else nullcontext() as res_f:
            file_paths = list(p.glob("*.jsonl")) if p.is_dir() else [p]

            if args.cont:
                finished_requests = load_requests_ids(args.results)

            if args.reverse:
                file_paths = reversed(file_paths)

            for f in tqdm(file_paths, desc="Processing split files", unit="file"):
                if finished_requests:
                    batch_ids = load_requests_ids(f)
                    if batch_ids.issubset(finished_requests):
                        continue

                if args.synchronous:
                    for result in api.process_request_file(f):
                        if finished_requests and result["custom_id"] in finished_requests:
                            continue
                        res_f.write(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + "\n")
                else:
                    res = api.batch_request_and_wait(f)

                    if finished_requests and len(batch_ids & finished_requests) > 0:
                        # remove already processed requests
                        for line in res.splitlines():
                            if json.loads(line)["custom_id"] in finished_requests:
                                continue
                            res_f.write(line+"\n")
                    else:
                        res_f.write(res)
            return

    print(json.dumps(api.process_line(args.file, args.line), ensure_ascii=False, separators=(',', ':')))


def print_histogram(data: Sequence[int], bins: int = 10, max_bars: int = 10, line_prefix="\t"):
    counts, bin_edges = np.histogram(data, bins=bins)
    pdf_counts = counts / sum(counts)

    for i in range(len(counts)):
        bin_range = f"{bin_edges[i]:.1f} - {bin_edges[i + 1]:.1f}"
        bar = "█" * int(pdf_counts[i] * max_bars)
        print(f"{line_prefix}{bin_range} | {bar} ({counts[i]} [{pdf_counts[i] * 100:.2f}%])")


def print_batch_stats(data: Sequence[int], line_prefix="\t"):
    """
    Prints statistics about the batch file.

    :param data: token counts per sample
    :param line_prefix: prefix for each line printed
    """

    print(f"{line_prefix}Number of samples: {len(data)}")
    print(f"{line_prefix}Number of tokens: {sum(data)}")
    print(f"{line_prefix}Min number of tokens: {min(data)}")
    print(f"{line_prefix}Max number of tokens: {max(data)}")
    mean = sum(data) / len(data)
    print(f"{line_prefix}Average number of tokens: {mean}")
    print(
        f"{line_prefix}Standard deviation of tokens: {(sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5}")

    print(f"{line_prefix}Histogram of number of tokens")
    print_histogram(data, bins=10, max_bars=10, line_prefix=line_prefix)


def batch_request_tokens(args):
    """
    Provides statistics about the number of tokens in the batch file.

    :param args: Parsed arguments.
    """
    tokenizers = {}
    number_of_tokens = []
    number_of_tokens_messages = []
    with open(args.file, mode='r') as f:
        for line in f:
            record = json.loads(line)
            model = record["body"]["model"]
            model = model.rstrip("-mini")
            if model not in tokenizers:
                tokenizers[model] = tiktoken.encoding_for_model(model)

            number_of_tokens_sample = 0
            for message in record["body"]["messages"]:
                token_cnt = len(tokenizers[model].encode(message["content"], allowed_special="all"))
                number_of_tokens_sample += token_cnt
                number_of_tokens_messages.append(token_cnt)

            number_of_tokens.append(number_of_tokens_sample)

    print("Granularity: sample")
    print_batch_stats(number_of_tokens)

    print("Granularity: message")
    print_batch_stats(number_of_tokens_messages)


def calculate_image_tokens(width: int, height: int) -> int:
    """
    Calculates number of tokens for image for OpenAI API.
    Taken from: https://community.openai.com/t/how-do-i-calculate-image-tokens-in-gpt4-vision/492318/5

    :param width: Number of pixels in width.
    :param height: Number of pixels in height.
    :return: Number of tokens.
    """
    if width > 2048 or height > 2048:
        aspect_ratio = width / height
        if aspect_ratio > 1:
            width, height = 2048, int(2048 / aspect_ratio)
        else:
            width, height = int(2048 * aspect_ratio), 2048

    if width >= height and height > 768:
        width, height = int((768 / height) * width), 768
    elif height > width and width > 768:
        width, height = 768, int((768 / width) * height)

    tiles_width = ceil(width / 512)
    tiles_height = ceil(height / 512)
    total_tokens = 85 + 170 * (tiles_width * tiles_height)

    return total_tokens


def split_batch(args):
    """
    Splits batch file into smaller files.
    """

    tokenizers = {}
    lines_cache = []
    number_of_tokens = 0
    file_cnt = 0
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(args.file, mode='r') as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            model = record["body"]["model"]

            if model not in tokenizers:
                tokenizers[model] = tiktoken.encoding_for_model(model)

            token_cnt = 0
            for message in record["body"]["messages"]:
                if isinstance(message["content"], list):
                    # multi modal
                    for sub_message in message["content"]:
                        if sub_message["type"] == "image_url":
                            # load image
                            image = sub_message["image_url"]["url"]
                            # link of base64 image
                            if image.startswith("data:image/"):
                                image = image.split(",")[1]
                                # load base64 image
                                image = Image.open(BytesIO(base64.b64decode(image)))
                                token_cnt += calculate_image_tokens(image.width, image.height)
                            elif image.startswith("http"):
                                # download image
                                r = requests.get(image, stream=True)
                                if r.status_code == 200:
                                    image = Image.open(BytesIO(r.content))
                                    token_cnt += calculate_image_tokens(image.width, image.height)
                                else:
                                    raise ValueError(f"Failed to download image: {image} in record {record['custom_id']}")
                            else:
                                raise ValueError(f"Unknown image type: {image} in record {record['custom_id']}")
                        elif sub_message["type"] == "text":
                            token_cnt += len(tokenizers[model].encode(sub_message["text"], allowed_special="all"))
                        else:
                            raise ValueError(f"Unknown message type: {sub_message['type']} in record {record['custom_id']}")

                else:
                    token_cnt += len(tokenizers[model].encode(message["content"], allowed_special="all"))

            if number_of_tokens > 0 and number_of_tokens + token_cnt > args.max_tokens:
                with open(output_path / f"batch_{file_cnt}.jsonl", mode='w') as out:
                    out.writelines(lines_cache)
                lines_cache = []
                number_of_tokens = 0
                file_cnt += 1

            lines_cache.append(line)
            number_of_tokens += token_cnt

        if len(lines_cache) > 0:
            with open(output_path / f"batch_{file_cnt}.jsonl", mode='w') as out:
                out.writelines(lines_cache)


def create_empty_config_for_create_batch_workflow(args, f: TextIO):
    """
    Creates empty configurations for CreateBatchWorkflow.

    :param args: Parsed arguments.
    :param f: File to write the configuration to.
    """
    print("creating config for CreateBatchWorkflow")

    convertor_subclasses = [c.__name__ for c in subclasses(Convertor)]
    convertor = inquirer.prompt([
        inquirer.List('convertor',
                      message="Choose convertor",
                      choices=convertor_subclasses,
                      )
    ])["convertor"]

    loader_subclasses = [c.__name__ for c in subclasses(Loader)]
    loader = inquirer.prompt([
        inquirer.List('loader',
                      message="Choose loader",
                      choices=loader_subclasses,
                      )
    ])["loader"]

    assembler_subclasses = [c.__name__ for c in subclasses(APISampleAssembler)]
    assembler = inquirer.prompt([
        inquirer.List('assembler',
                      message="Choose sample assembler",
                      choices=assembler_subclasses,
                      )
    ])["assembler"]

    template = None

    if assembler in set(c.__name__ for c in subclasses(TemplateBasedAssembler)):
        template_subclasses = [c.__name__ for c in subclasses(Template)]
        template = inquirer.prompt([
            inquirer.List('template',
                          message="Choose template type",
                          choices=template_subclasses,
                          )
        ])["template"]

    save_to = StringIO()

    config = Config(CreateBatchWorkflow,
                    file_override_user_defaults={
                        "convertor": {
                            "cls": convertor,
                            "config": {
                                "loader": {
                                    "cls": loader,
                                    "config": {}
                                },
                                "sample_assembler": {
                                    "cls": assembler,
                                    "config": {
                                        "input_template": {
                                            "cls": template,
                                            "config": {}
                                        }
                                    } if template is not None else {}
                                }
                            }
                        }
                    })

    config.save(save_to)

    f.write(save_to.getvalue())


def create_empty_config_for_api(args, f: TextIO):
    """
    Creates empty configurations for API.

    :param args: Parsed arguments.
    :param f: File to write the configuration to.
    """
    print("creating config for API")
    conv_subclasses = [c.__name__ for c in subclasses(API)]
    api = inquirer.prompt([
        inquirer.List('api',
                      message="Choose api",
                      choices=conv_subclasses,
                      )
    ])["api"]

    save_to = StringIO()

    config = Config(APISelector,
                    file_override_user_defaults={
                        "api": {
                            "cls": api,
                            "config": {}
                        }
                    })

    config.save(save_to)

    f.write(save_to.getvalue())


def create_config(args):
    """
    Creates empty configurations and saves them to semantsumannot/configs.

    :param args: Parsed arguments.
    """

    config_type = inquirer.prompt([
        inquirer.List('config_type',
                      message="Which configuration do you want to create?",
                      choices=["create batch workflow", "API"],
                      )
    ])["config_type"]

    with (sys.stdout if args.path is None else open(args.path, mode='w')) as f:
        if config_type == "create batch workflow":
            create_empty_config_for_create_batch_workflow(args, f)
        elif config_type == "API":
            create_empty_config_for_api(args, f)
        else:
            raise ValueError(f"Unknown configuration type: {config_type}")


def prompt_res_pair(args):
    """
    Pairs prompt with response.

    :param args: Parsed arguments.
    """

    with open(args.prompts, mode='r') as prompt_file, open(args.response, mode='r') as response_file:
        # read response ids
        id_2_response_file_offset = {}
        json_fields = args.json if args.json is not None else None
        with tqdm(desc="Indexing response file", unit="bytes", total=os.path.getsize(args.response)) as pbar:
            while line := response_file.readline():
                record = json.loads(line)
                if json_fields is not None:
                    content = record["response"]["body"]["choices"][0]["message"]["content"].strip()

                    if args.skip:
                        try:
                            response_fields = json.loads(content)
                        except JSONDecodeError:
                            continue
                    else:
                        response_fields = read_potentially_malformed_json_result(content)

                    if any(f not in response_fields for f in json_fields):
                        raise ValueError(f"Fields {json_fields} not found in record {record['custom_id']}. The content is:\n{content}\nParsed response fields: {response_fields}")
                id_2_response_file_offset[record["custom_id"]] = pbar.n
                pbar.update(response_file.tell() - pbar.n)

        fieldnames = ["prompt"]
        if json_fields is None:
            fieldnames.append("response")
        else:
            fieldnames.extend(json_fields)

        dict_writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        dict_writer.writeheader()
        for prompt_line in prompt_file:
            prompt = json.loads(prompt_line)

            if (args.missing or args.skip) and prompt["custom_id"] not in id_2_response_file_offset:
                # skip missing records
                continue

            response_file.seek(id_2_response_file_offset[prompt["custom_id"]])
            response = json.loads(response_file.readline())

            if json_fields is not None:
                parsed_json = read_potentially_malformed_json_result(response["response"]["body"]["choices"][0]["message"]["content"])
                res = {f: parsed_json[f] for f in json_fields}
                res["prompt"] = prompt["body"]["messages"][0]["content"]
            else:
                res = {
                    "prompt": prompt["body"]["messages"][0]["content"],
                    "response": response["response"]["body"]["choices"][0]["message"]["content"]
                }
            dict_writer.writerow(res)


def main():
    logging.basicConfig(format='%(process)d: %(levelname)s : %(asctime)s : %(message)s', level=logging.WARNING)

    parser = ArgumentParser(description="OpenAI API caller.")
    subparsers = parser.add_subparsers()

    ask_parser = subparsers.add_parser("create_batch_file",
                                       help="Creates batch file for OpenAI API. Results are printed to stdout.")
    ask_parser.add_argument("--path", help="Path to data.", type=str, default=None)
    ask_parser.add_argument("-c", "--config", help="Path to the configuration file.", default=None)
    ask_parser.set_defaults(func=create_batch_file)

    batch_request_parser = subparsers.add_parser("batch_request", help="Sends requests to OpenAI API.")
    batch_request_parser.add_argument("file", help="Path to the batch file.")
    batch_request_parser.add_argument("-c", "--config", help="Path to the configuration file.", default=None)
    batch_request_parser.add_argument("-l", "--line",
                                      help="This allows to send only one request on given line number (starts from 0) from the batch file. Warning, this will use standard request.",
                                      default=None, type=int)
    batch_request_parser.add_argument("-r", "--results", help="Path to the file where the results should be saved.",
                                      default=None)
    batch_request_parser.add_argument("-s", "--synchronous", help="Forces to use synchronous API instead of batch API.",
                                      action="store_true")
    batch_request_parser.add_argument("--reverse", help="Reverse the order of batch splits.", action="store_true")
    batch_request_parser.add_argument("--cont",
                                      help="Continue processing of the batch file. If the results file is specified, it will skip processing of completely processed batch files.",
                                      action="store_true")
    batch_request_parser.set_defaults(func=batch_request)

    batch_request_tokens_parser = subparsers.add_parser("batch_request_tokens",
                                                        help="Provides statistics about the number of tokens in the batch file, works only for openai models.")
    batch_request_tokens_parser.add_argument("file", help="Path to the batch file.")
    batch_request_tokens_parser.set_defaults(func=batch_request_tokens)

    split_batch_parser = subparsers.add_parser("split_batch", help="Splits batch file into smaller files.")
    split_batch_parser.add_argument("file", help="Path to the batch file.")
    split_batch_parser.add_argument("output", help="Path to the output folder.")
    split_batch_parser.add_argument("max_tokens", help="Maximum number of tokens in one file.", type=int)
    split_batch_parser.set_defaults(func=split_batch)

    prompt_res_pair_parser = subparsers.add_parser("prompt_res_pair", help="Pairs prompt with response.")
    prompt_res_pair_parser.add_argument("prompts", help="Path to the batch file with prompts.")
    prompt_res_pair_parser.add_argument("response", help="Path to the response file.")
    prompt_res_pair_parser.add_argument("-j", "--json",
                                        help="List of fields that should be taken from result that is in json format.",
                                        nargs="+",
                                        default=None)
    prompt_res_pair_parser.add_argument("-m", "--missing",
                                        help="Allows to skip records that are missing in the response file.",
                                        action="store_true")
    prompt_res_pair_parser.add_argument("-s", "--skip",
                                        help="Skip malformed records.", action="store_true")
    prompt_res_pair_parser.set_defaults(func=prompt_res_pair)

    create_config_parser = subparsers.add_parser("create_config", help="Creates configuration.")
    create_config_parser.add_argument("--path", help="Path to the configuration file.")
    create_config_parser.set_defaults(func=create_config)

    args = parser.parse_args()

    if args is not None:
        args.func(args)
    else:
        exit(1)


if __name__ == '__main__':
    main()
