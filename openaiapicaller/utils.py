# -*- coding: UTF-8 -*-
"""
Created on 02.12.24

:author:     Martin Dočekal
"""
import base64
import json
import re
import sys
import time
from json import JSONDecodeError

import json_repair
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
