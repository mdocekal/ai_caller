# -*- coding: UTF-8 -*-
"""
Created on 02.12.24

:author:     Martin Dočekal
"""
import base64
import json
from io import BytesIO
from json import JSONDecodeError
from math import ceil

import json_repair
import requests
import tiktoken
from PIL import Image


def jsonl_field_value_2_file_offset_mapping(file: str, field: str) -> dict:
    """
    Creates mapping of field value to file line offset.

    Field value after overwrites the previous value.

    :param file: Path to the file.
    :param field: Field name.
    :return: Mapping of field value to file line offset.
    """

    mapping = {}

    with open(file, "r") as f:
        offset = 0
        while line := f.readline():
            data = json.loads(line)
            mapping[data[field]] = offset
            offset = f.tell()

    return mapping


def read_potentially_malformed_json_result(j: str, should_be_dict: bool = True) -> dict:
    """
    Reads JSON string that is potentially malformed.

    :param j: JSON string.
    :param should_be_dict: If True the JSON should be dictionary.
    :return: parsed JSON.
    :raises JSONDecodeError: If the JSON is not parsable.
    """

    r = json_repair.loads(j)
    if should_be_dict and not isinstance(r, dict):
        if not isinstance(r[0], dict):
            raise JSONDecodeError("Could not parse JSON.", j, 0)
        r = r[0]
    return r


def obtain_base64_image(path: str) -> str:
    """
    Obtains base64 image from the path.

    :param path: path to the image
    :return: base64 image
    """

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def is_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def detect_image_format(path: str) -> str:
    """
    Detects image format from the path.

    :param path: path to the image
    :return: image format
    """
    with Image.open(path) as img:
        return img.format.lower()


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

OPENAI_LOW_TOKENS = 85

class TokenCounter:
    """
    Token counter for OpenAI API.
    """

    def __init__(self):
        self.tokenizers = {}
        self.token_count = 0

    def __call__(self, sample: dict) -> int:
        """
        Tokenizes the sample and counts the tokens.

        :param sample: Batch sample to tokenize.
        :return: number of tokens in sample
        """

        model = sample["body"]["model"]

        if model not in self.tokenizers:
            self.tokenizers[model] = tiktoken.encoding_for_model(model)

        token_cnt = 0
        for message in sample["body"]["messages"]:
            if isinstance(message["content"], list):
                # multi modal
                for sub_message in message["content"]:
                    if sub_message["type"] == "image_url":
                        if "detail" in sub_message["image_url"] and sub_message["image_url"]["detail"].lower() == "low":
                            token_cnt += OPENAI_LOW_TOKENS
                        else:
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
                                    raise ValueError(f"Failed to download image: {image} in record {sample['custom_id']}")
                            else:
                                raise ValueError(f"Unknown image type: {image} in record {sample['custom_id']}")
                    elif sub_message["type"] == "text":
                        token_cnt += len(self.tokenizers[model].encode(sub_message["text"], allowed_special="all"))
                    else:
                        raise ValueError(f"Unknown message type: {sub_message['type']} in record {sample['custom_id']}")

            else:
                token_cnt += len(self.tokenizers[model].encode(message["content"], allowed_special="all"))

        self.token_count += token_cnt
        return token_cnt


