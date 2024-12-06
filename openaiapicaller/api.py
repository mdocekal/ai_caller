import json
import sys
import time
from typing import Optional, Union

from classconfig import ConfigurableValue
from classconfig.validators import StringValidator
from openai import OpenAI, APIError, RateLimitError
from openai.types.batch import Batch
from tqdm import tqdm


class API:
    """
    Handles requests to the API.
    """

    api_key: str = ConfigurableValue(desc="OpenAI API key.", validator=StringValidator())
    pool_interval: Optional[int] = ConfigurableValue(
        desc="Interval in seconds for checking the status of the batch request.",
        user_default=300,
        voluntary=True,
        validator=lambda x: x is None or x > 0)
    process_request_file_interval: Optional[int] = ConfigurableValue(
        desc="Interval in seconds between sending requests in process_request_file.",
        user_default=1,
        voluntary=True,
        validator=lambda x: x is None or x > 0)
    base_url: Optional[str] = ConfigurableValue(desc="Base URL for OpenAI API.", user_default=None, voluntary=True)

    def __init__(self, api_key: str, pool_interval: int = 300, base_url: Optional[str] = None, process_request_file_interval: int = 1):
        self.api_key = api_key
        self.pool_interval = pool_interval
        self.process_request_file_interval = process_request_file_interval
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def process_request_file(self, path_to_file: str, serialized: bool = True) -> Union[dict, str]:
        """
        Simulates the batch request, but uses normal synchronous API calls.

        :param path_to_file: Path to the file with requests.
        :param serialized: If True, the output will be serialized to a string.
            It is True by default to be compatible with the batch request.
        :return: Content of the output file
        """

        res = []

        with open(path_to_file, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(tqdm(lines, desc="Sending requests")):
            record = json.loads(line)
            if len(res) > 0:
                time.sleep(self.process_request_file_interval)

            while True:
                try:
                    response = self.client.chat.completions.create(**record["body"])
                    break
                except RateLimitError:
                    print(f"Rate limit reached. Waiting for {self.pool_interval} seconds.", flush=True,
                          file=sys.stderr)
                    time.sleep(self.pool_interval)

            res.append({
                "id": i,
                "custom_id": record["custom_id"],
                "response": {
                    "body": response.model_dump()
                },
                "error": None
            })

        if serialized:
            res = "\n".join(json.dumps(js) for js in res)
            res += "\n"

        return res

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
                if any("Enqueued token limit reached for" in err.message for err in e.response.errors.data):
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

        raise APIError("Batch request failed with status: " + batch.status, response=batch)


