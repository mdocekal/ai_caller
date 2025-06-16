import csv
import json
import logging
import os.path
import sys
from argparse import ArgumentParser
from contextlib import nullcontext
from io import StringIO
from json import JSONDecodeError
from pathlib import Path
from typing import Sequence, Optional, TextIO

import inquirer
import numpy as np
import tiktoken
from classconfig import Config, ConfigurableFactory, ConfigurableMixin, ConfigurableSubclassFactory, ConfigurableValue
from classconfig.classes import subclasses, sub_cls_from_its_name
from tqdm import tqdm

from aicaller.api import API
from aicaller.conversion import Convertor
from aicaller.loader import Loader
from aicaller.sample_assembler import APISampleAssembler, TemplateBasedAssembler
from aicaller.template import Template, StringTemplate
from aicaller.utils import read_potentially_malformed_json_result, TokenCounter


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


class InputTemplateConfig(ConfigurableMixin):
    """
    Configuration for input template.
    """

    input_template: Template = ConfigurableSubclassFactory(Template, "Template for input assembly.",
                                                           user_default=StringTemplate)


def create_batch_file(args):
    """
    Creates batch file for OpenAI API. Results are printed to stdout.

    :param args: Parsed arguments.
    """

    config_path = Path(args.config)
    config = Config(CreateBatchWorkflow).load(config_path)
    if args.input_template_config is not None:
        # check whether given convertor allows to override input_template
        if "sample_assembler" not in config["convertor"]["config"]:
            raise ValueError("The convertor does not support overriding input_template.")

        try:
            _ = sub_cls_from_its_name(TemplateBasedAssembler, config["convertor"]["config"]["sample_assembler"]["cls"])
        except ValueError:
            raise ValueError("The convertor does not support overriding input_template.")

        # load input template from the given file
        input_config_path = Path(args.input_template_config)
        input_template_config = Config(InputTemplateConfig).load(input_config_path)

        config["convertor"]["config"]["sample_assembler"]["config"]["input_template"]["cls"] = input_template_config["input_template"]["cls"]
        config["convertor"]["config"]["sample_assembler"]["config"]["input_template"]["config"] = input_template_config["input_template"]["config"]

        config.untransformed["convertor"]["config"]["sample_assembler"]["config"]["input_template"]["cls"] = input_template_config.untransformed["input_template"]["cls"]
        config.untransformed["convertor"]["config"]["sample_assembler"]["config"]["input_template"]["config"] = input_template_config.untransformed["input_template"]["config"]

    workflow = ConfigurableFactory(CreateBatchWorkflow).create(config)
    workflow(proc_path=args.path)


def load_requests_ids(p: str, expected_ids: None | set[str] = None) -> set[str]:
    """
    Loads request ids from the file or directory.

    :param p: Path to the file or directory
    :param expected_ids: Set of expected ids is used for getting ids from directory.
    :return: Set of request ids.
    """

    if p.endswith(os.sep) or os.altsep and p.endswith(os.altsep):
        if expected_ids is None:
            raise ValueError("Expected ids must be provided when processing directory.")
        p = Path(p)
        return {i for i in expected_ids if (p / f"{i}.jsonl").exists() or (p / f"{i}.txt").exists()}

    else:
        with open(p, mode='r') as f:
            return {json.loads(l)["custom_id"] for l in f}


class APISelector(ConfigurableMixin):
    api: API = ConfigurableSubclassFactory(API, "API type")


def batch_request(args):
    """
    Sends requests to OpenAI API.

    :param args: Parsed arguments.
    """
    config_path = Path(args.config)
    config = Config(APISelector).load(config_path)
    api = ConfigurableFactory(APISelector).create(config).api

    if args.line is not None:
        print(json.dumps(api.process_line(args.file, args.line), ensure_ascii=False, separators=(',', ':')))
        return

    if args.results is None:
        raise ValueError("Results file must be specified when sending batch requests.")

    p = Path(args.file)

    res_dir = None
    if args.results is not None and args.results.endswith(os.sep) or os.altsep and args.results.endswith(os.altsep):
        res_dir = Path(args.results)
        if not res_dir.exists():
            res_dir.mkdir(parents=True)

    with open(args.results, mode=('a' if args.cont else 'w')) if args.results is not None and res_dir is None else nullcontext() as res_f:
        if args.results is None:
            res_f = sys.stdout

        file_paths = list(p.glob("*.jsonl")) if p.is_dir() else [p]

        # Load finished requests ids if continuing
        finished_requests = set()

        if args.cont:
            if res_dir is not None:
                all_ids = set()
                for f in file_paths:
                    all_ids |= load_requests_ids(str(f))

                finished_requests = load_requests_ids(args.results, expected_ids=all_ids)
            else:
                finished_requests = load_requests_ids(args.results)

        if args.reverse:
            file_paths = reversed(file_paths)

        for f in tqdm(file_paths, desc="Processing split files", unit="file"):
            if finished_requests:
                batch_ids = load_requests_ids(str(f))
                if batch_ids.issubset(finished_requests):
                    continue

            res = api.process_request_file(str(f)) if args.synchronous else api.batch_request_and_wait(str(f))
            for api_output in res:
                if finished_requests and api_output.custom_id in finished_requests:
                    continue

                if args.only_output:
                    if res_dir is None:
                        result_to_write = json.dumps({
                            "custom_id": api_output.custom_id,
                            "content": api_output.response.get_raw_content(),
                        }, ensure_ascii=False, separators=(',', ':'))
                    else:
                        result_to_write = api_output.response.get_raw_content()
                else:
                    result_to_write = api_output.model_dump_json()

                if res_dir is None:
                    print(result_to_write, file=res_f)
                else:
                    suffix = ".json" if api_output.response.structured else ".txt"
                    with open(res_dir / (api_output.custom_id + suffix), "w") as res_f_separate:
                        print(result_to_write, file=res_f_separate)


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


def batch_stats(args):
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


def batch_tokens(args):
    """
    Counts tokens in batch file.
    """

    token_counter = TokenCounter()
    with open(args.file, mode='r') as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            token_counter(record)

    print(token_counter.token_count)


def split_batch(args):
    """
    Splits batch file into smaller files.
    """

    lines_cache = []
    number_of_tokens = 0
    file_cnt = 0
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    token_counter = TokenCounter()
    with open(args.file, mode='r') as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            token_cnt = token_counter(record)

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
    ask_parser.add_argument("-c", "--config", help="Path to the configuration file.")
    ask_parser.add_argument("-i", "--input_template_config", help="Path to the input prompt configuration file. This will load input_template from separate file and overwrite the one in the config. Might not be available for all convertors.",
                            type=str, default=None)
    ask_parser.add_argument("--path", help="Path to data.", type=str, default=None)
    ask_parser.set_defaults(func=create_batch_file)

    split_batch_parser = subparsers.add_parser("split_batch", help="Splits batch file into smaller files.")
    split_batch_parser.add_argument("file", help="Path to the batch file.")
    split_batch_parser.add_argument("output", help="Path to the output folder.")
    split_batch_parser.add_argument("max_tokens", help="Maximum number of tokens in one file.", type=int)
    split_batch_parser.set_defaults(func=split_batch)

    batch_request_parser = subparsers.add_parser("batch_request", help="Sends requests to OpenAI API.")
    batch_request_parser.add_argument("file", help="Path to the batch file or directory with batch files.")
    batch_request_parser.add_argument("-c", "--config", help="Path to the API configuration file.")
    batch_request_parser.add_argument("-l", "--line",
                                      help="This allows to send only one request on given line number (starts from 0) from the batch file. Warning, this will use standard request.",
                                      default=None, type=int)
    batch_request_parser.add_argument("-r", "--results", help="Path to the file where the results should be saved. If directory it will create a separate file for each request with the same name as custom_id. It wil use extension .txt for simple output and .json for structured output.",
                                      default=None)
    batch_request_parser.add_argument("-s", "--synchronous", help="Forces to use synchronous API instead of batch API. Calls are always synchronous for Ollama API.",
                                      action="store_true")
    batch_request_parser.add_argument("--reverse", help="Reverse the order of batch splits.", action="store_true")
    batch_request_parser.add_argument("--cont",
                                      help="Continue processing of the batch file. If the results file is specified, it will skip processing of completely processed batch files.",
                                      action="store_true")
    batch_request_parser.add_argument("--only_output",
                                        help="If specified, will write only model output (0. choice) to results file, without metadata.",
                                        action="store_true")
    batch_request_parser.set_defaults(func=batch_request)

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

    batch_request_tokens_parser = subparsers.add_parser("batch_stats",
                                                        help="Provides statistics about the number of tokens in the batch file, works only for openai models.")
    batch_request_tokens_parser.add_argument("file", help="Path to the batch file.")
    batch_request_tokens_parser.set_defaults(func=batch_stats)

    count_tokens_batch_parser = subparsers.add_parser("batch_tokens", help="Counts tokens in batch file, works only for openai models.")
    count_tokens_batch_parser.add_argument("file", help="Path to the batch file.")
    count_tokens_batch_parser.set_defaults(func=batch_tokens)

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
