import json
import sys
import time
from abc import ABC, abstractmethod
from typing import Optional, Generator

from classconfig import ConfigurableValue
from classconfig.validators import StringValidator
from ollama import Client as OllamaClient
from openai import OpenAI, APIError, RateLimitError
from openai.types.batch import Batch
from tqdm import tqdm


class API(ABC):
    """
    Handles requests to the API.
    """

    api_key: str = ConfigurableValue(desc="API key.", validator=StringValidator())
    pool_interval: Optional[int] = ConfigurableValue(
        desc="Interval in seconds for checking the status of the batch request.",
        user_default=300,
        voluntary=True,
        validator=lambda x: x is None or x > 0)
    process_request_file_interval: Optional[int] = ConfigurableValue(
        desc="Interval in seconds between sending requests in process_request_file.",
        user_default=1,
        voluntary=True,
        validator=lambda x: x is None or x >= 0)
    base_url: Optional[str] = ConfigurableValue(desc="Base URL for API.", user_default=None, voluntary=True)

    def __init__(self, api_key: str, pool_interval: int = 300, base_url: Optional[str] = None, process_request_file_interval: int = 1):
        self.api_key = api_key
        self.pool_interval = pool_interval
        self.process_request_file_interval = process_request_file_interval
        self.base_url = base_url

    @abstractmethod
    def process_line(self, path_to_file: str, line: int) -> dict:
        """
        Processes a line from the file.

        :param path_to_file: Path to the file with requests.
        :param line: Line number.
        :return: Processed line
        """
        ...

    @abstractmethod
    def process_request_file(self, path_to_file: str) -> Generator[dict, None, None]:
        """
        Simulates the batch request, but uses normal synchronous API calls.

        :param path_to_file: Path to the file with requests.
        :return: Results for each request
        """
        ...

    @abstractmethod
    def batch_request(self, path_to_file: str) -> dict:
        """
        Sends requests to API.

        :param path_to_file: Path to the file with requests.
        :return: Batch request response
        """
        ...

    @abstractmethod
    def batch_request_and_wait(self, path_to_file: str) -> str:
        """
        Sends requests to API and waits for the batch request to finish.

        In case it receives an error that the enqueued token limit was reached, it will wait for the pool_interval
        and try again.

        :param path_to_file: Path to the file with requests.
        :return: Content of the output file if the batch request was successful.
        :raises APIError: If the batch request failed.
        """
        ...

    @abstractmethod
    def wait_for_batch_request(self, response: Batch) -> str:
        """
        Waits for the batch request to finish and downloads the results.

        :param response: Batch request response from API.
        :return: Content of the output file if the batch request was successful.
        :raises APIError: If the batch request failed.
        """
        ...


class OpenAPI(API):
    """
    Handles requests to the OpenAI API.
    """

    def __init__(self, api_key: str, pool_interval: int = 300, base_url: Optional[str] = None, process_request_file_interval: int = 1):
        super().__init__(api_key, pool_interval, base_url, process_request_file_interval)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def process_line(self, path_to_file: str, line: int) -> dict:
        with open(path_to_file, mode='r') as f:
            for i, l in enumerate(f):
                if i == line:
                    line = l
                    break
            else:
                raise ValueError(f"Line {line} not found.")
            body = json.loads(line)["body"]

        response = self.client.chat.completions.create(**body)

        return {
            "custom_id": json.loads(line)["custom_id"],
            "response": {
                "body": response.model_dump()
            },
            "error": None
        }

    def process_request_file(self, path_to_file: str) -> Generator[dict, None, None]:
        with open(path_to_file, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(tqdm(lines, desc="Sending requests")):
            record = json.loads(line)
            if i > 0 and self.process_request_file_interval > 0:
                time.sleep(self.process_request_file_interval)

            while True:
                try:
                    response = self.client.chat.completions.create(**record["body"])
                    break
                except RateLimitError:
                    print(f"Rate limit reached. Waiting for {self.pool_interval} seconds.", flush=True,
                          file=sys.stderr)
                    time.sleep(self.pool_interval)

            yield {
                "custom_id": record["custom_id"],
                "response": {
                    "body": response.model_dump()
                },
                "error": None
            }

    def batch_request(self, path_to_file: str) -> dict:
        """
        Sends requests to OpenAI API.

        :param path_to_file: Path to the file with requests.
        :return: Batch request response
        """

        batch_input_file = self.client.files.create(
            file=open(path_to_file, "rb"),
            purpose="batch"
        )
        return self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

    def batch_request_and_wait(self, path_to_file: str) -> str:
        """
        Sends requests to OpenAI API and waits for the batch request to finish.

        In case it receives an error that the enqueued token limit was reached, it will wait for the pool_interval
        and try again.

        :param path_to_file: Path to the file with requests.
        :return: Content of the output file if the batch request was successful.
        :raises APIError: If the batch request failed.
        """

        while True:
            try:
                response = self.batch_request(path_to_file)
                return self.wait_for_batch_request(response)
            except APIError as e:
                if any("Enqueued token limit reached for" in err.message for err in e.body.errors.data):
                    print("Enqueued token limit reached. Waiting for the pool interval.", flush=True,
                          file=sys.stderr)
                    time.sleep(self.pool_interval)

    def wait_for_batch_request(self, response: Batch) -> str:
        """
        Waits for the batch request to finish and downloads the results.

        :param response: Batch request response from OpenAI API.
        :return: Content of the output file if the batch request was successful.
        :raises APIError: If the batch request failed.
        """

        batch_id = response.id
        while True:
            batch: Batch = self.client.batches.retrieve(batch_id)
            if batch.status == "completed":
                break
            if batch.status in {"failed", "canceled", "expired"}:
                break
            time.sleep(self.pool_interval)

        if batch.status == "completed":
            file_response = self.client.files.content(batch.output_file_id)
            return file_response.text

        raise APIError("Batch request failed with status: " + batch.status, None, body=batch)


class OllamaAPI(API):

    def __init__(self, api_key: str, pool_interval: int = 300, base_url: Optional[str] = None, process_request_file_interval: int = 1):
        super().__init__(api_key, pool_interval, base_url, process_request_file_interval)
        self.client = OllamaClient(host=base_url)

    def process_line(self, path_to_file: str, line: int) -> dict:
        with open(path_to_file, mode='r') as f:
            for i, l in enumerate(f):
                if i == line:
                    line = l
                    break
            else:
                raise ValueError(f"Line {line} not found.")
            body = json.loads(line)["body"]

        response = self.client.chat(**body)

        return {
            "custom_id": json.loads(line)["custom_id"],
            "response": {
                "body": response.model_dump()
            },
            "error": None
        }

    def process_request_file(self, path_to_file: str) -> Generator[dict, None, None]:
        with open(path_to_file, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(tqdm(lines, desc="Sending requests")):
            record = json.loads(line)
            if i > 0 and self.process_request_file_interval > 0:
                time.sleep(self.process_request_file_interval)

            response = self.client.chat(**record["body"])

            yield {
                "custom_id": record["custom_id"],
                "response": {
                    "body": response.model_dump()
                },
                "error": None
            }

    def batch_request(self, path_to_file: str) -> dict:
        raise NotImplementedError("Batch request is not supported by Ollama API.")

    def batch_request_and_wait(self, path_to_file: str) -> str:
        raise NotImplementedError("Batch request is not supported by Ollama API.")

    def wait_for_batch_request(self, response: Batch) -> str:
        raise NotImplementedError("Batch request is not supported by Ollama API.")

