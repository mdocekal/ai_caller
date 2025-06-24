import json
import sys
import time
from abc import abstractmethod
from collections.abc import Iterable
from typing import Generator, Container, Optional

from ollama import Client as OllamaClient
from openai import OpenAI, APIError, RateLimitError
from openai.types.batch import Batch

from aicaller.api import APIOutput, APIResponseOpenAI, APIResponseOllama
from aicaller.api.base import APIBase, APIRequest


class API(APIBase):
    """
    Handles requests to the API.
    """

    @abstractmethod
    def process_single_request(self, request: APIRequest) -> APIOutput:
        """
        Processes a single request.

        :param request: Request dictionary.
        :return: Processed request
        """
        ...

    def process_requests(self, requests: Iterable[APIRequest]) -> Generator[APIOutput, None, None]:
        """
        Processes a list of requests.

        :param requests: Iterable of request dictionaries.
        :return: Processed requests
        """
        for i, request in enumerate(requests):
            if i > 0 and self.process_requests_interval > 0:
                time.sleep(self.process_requests_interval)
            yield self.process_single_request(request)

    def process_line(self, path_to_file: str, line: int) -> APIOutput:
        """
        Processes a line from the file.

        :param path_to_file: Path to the file with requests.
        :param line: Line number.
        :return: Processed line
        :raises ValueError: If the line is not found.
        :raises ValidationError: If the line is not a valid request.
        """
        with open(path_to_file, mode='r') as f:
            for i, l in enumerate(f):
                if i == line:
                    line = l
                    break
            else:
                raise ValueError(f"Line {line} not found.")
            record = APIRequest.model_validate_json(line.strip())

        return self.process_single_request(record)

    @staticmethod
    def read_request_file(path_to_file: str) -> Iterable[APIRequest]:
        """
        Reads requests from a file.

        :param path_to_file: Path to the file with requests.
        :return: Iterable of requests
        :raises ValueError: If the file is empty or not found.
        :raises ValidationError: If the file contains invalid requests.
        """
        with open(path_to_file, "r") as f:
            for line in f:
                yield APIRequest.model_validate_json(line)

    def process_request_file(self, path_to_file: str, skip: Optional[Container[str]] = None) -> Generator[APIOutput, None, None]:
        """
        Simulates the batch request, but uses normal synchronous API calls.

        :param path_to_file: Path to the file with requests.
        :param skip: Set of custom request IDs to skip.
        :return: Results for each request
        """
        for i, record in enumerate(self.read_request_file(path_to_file)):
            if skip is not None and record.custom_id in skip:
                continue

            if i > 0 and self.process_requests_interval > 0:
                time.sleep(self.process_requests_interval)

            yield self.process_single_request(record)

    @abstractmethod
    def batch_request(self, path_to_file: str) -> dict:
        """
        Sends requests to API.

        :param path_to_file: Path to the file with requests.
        :return: Batch request response
        """
        ...

    @abstractmethod
    def batch_request_and_wait(self, path_to_file: str) -> list[APIOutput]:
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

    def __post_init__(self):
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def process_single_request(self, request: APIRequest) -> APIOutput:
        try:
            while True:
                try:
                    response = self.client.chat.completions.create(**request.body.model_dump(exclude={"type"}))
                    break
                except RateLimitError:
                    print(f"Rate limit reached. Waiting for {self.pool_interval} seconds.", flush=True,
                          file=sys.stderr)
                    time.sleep(self.pool_interval)

            return APIOutput(
                custom_id=request.custom_id,
                response=APIResponseOpenAI(
                    body=response.model_dump(),
                    structured=request.body.structured
                ),
                error=None
            )
        except APIError as e:
            return APIOutput(
                custom_id=request.custom_id,
                response=None,
                error=str(e)
            )

    def batch_request(self, path_to_file: str) -> Batch:
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

    def read_batch_file(self, path_to_file: str) -> dict[APIRequest]:
        """
        Reads requests from a batch file.

        :param path_to_file: Path to the file with requests.
        :return: Dictionary of requests indexed by custom_id
        """

        with open(path_to_file, "r") as f:
            samples = {}
            for line in f:
                record = APIRequest.model_validate_json(line.strip())
                if record.custom_id in samples:
                    raise ValueError(f"Duplicate custom_id found: {record.custom_id}")
                samples[record.custom_id] = record
        return samples

    def batch_request_and_wait(self, path_to_file: str) -> list[APIOutput]:
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
                batch_samples = self.read_batch_file(path_to_file)
                response = self.batch_request(path_to_file)
                file_content = self.wait_for_batch_request(response)
                content = []

                for line in file_content.splitlines():
                    record = json.loads(line)
                    content.append(APIOutput(
                        custom_id=record["custom_id"],
                        response=APIResponseOpenAI(
                            body=record["response"]["body"],
                            structured=batch_samples[record["custom_id"]].body.structured
                        ),
                        error=None
                    ))

                    return content
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

    def __post_init__(self):
        self.client = OllamaClient(host=self.base_url)

    def process_single_request(self, request: APIRequest) -> APIOutput:
        try:
            response = self.client.chat(**request.body.model_dump(exclude={"type"}))

            return APIOutput(
                custom_id=request.custom_id,
                response=APIResponseOllama(
                    body=response.model_dump(),
                    structured=request.body.structured
                ),
                error=None
            )
        except Exception as e:
            return APIOutput(
                custom_id=request.custom_id,
                response=None,
                error=str(e)
            )

    def batch_request(self, path_to_file: str) -> dict:
        raise NotImplementedError("Batch request is not supported by Ollama API.")

    def batch_request_and_wait(self, path_to_file: str) -> str:
        raise NotImplementedError("Batch request is not supported by Ollama API.")

    def wait_for_batch_request(self, response: Batch) -> str:
        raise NotImplementedError("Batch request is not supported by Ollama API.")
