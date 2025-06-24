import asyncio
import queue
import sys
import threading
import time
from abc import abstractmethod
from asyncio import as_completed
from collections.abc import Iterable
from typing import Container, AsyncGenerator, Optional, Generator

from ollama import AsyncClient
from openai import APIError, RateLimitError, AsyncOpenAI

from aicaller.api import APIOutput
from aicaller.api.base import APIResponseOllama, APIResponseOpenAI, APIBase, APIRequest


class APIAsync(APIBase):
    """
    Handles asynchronous requests to the API.
    """

    @abstractmethod
    async def process_single_request(self, request: APIRequest) -> APIOutput:
        """
        Processes a single request.

        :param request: Request dictionary.
        :return: Processed request
        """
        ...

    async def process_requests(self, requests: Iterable[APIRequest]) -> AsyncGenerator[APIOutput, None]:
        """
        Processes a list of requests.

        :param requests: Iterable of request dictionaries.
        :return: Processed requests
        """
        for o in as_completed(self.process_single_request(request) for request in requests):
            yield await o

    @staticmethod
    def read_request_file(path_to_file: str) -> Iterable[APIRequest]:
        """
        Reads requests from a file.

        :param path_to_file: Path to the file with requests.
        :return: Iterable of request dictionaries
        """
        with open(path_to_file, "r") as f:
            for line in f:
                yield APIRequest.model_validate_json(line)

    def process_request_file(self, path_to_file: str, skip: Optional[Container[str]] = None) -> Generator[APIOutput, None, None]:
        """
        Processes requests from a file, skipping those with IDs in the skip set.
        It works like a bridge between synchronous and asynchronous processing.

        :param path_to_file: Path to the file with requests.
        :param skip: Set of custom request IDs to skip.
        :return: Results for each request
        """

        q = queue.Queue()
        sentinel = object()

        def runner():
            """
            Runner function to process requests in a separate thread.
            """

            async def produce():
                try:
                    async for o in self.process_requests(
                            request for request in self.read_request_file(path_to_file)
                            if skip is None or request.custom_id not in skip
                    ):
                        q.put(o)
                finally:
                    q.put(sentinel)

            asyncio.run(produce())

        thread = threading.Thread(target=runner)
        thread.start()
        try:
            while True:
                item = q.get()
                if item is sentinel:
                    break
                yield item
        finally:
            thread.join()


class OpenAsyncAPI(APIAsync):
    """
    Handles asynchronous requests to the OpenAI API.
    """

    def __post_init__(self):
        self.client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url
        )
        self.semaphore = asyncio.Semaphore(self.concurrency)

    async def process_single_request(self, request: APIRequest) -> APIOutput:
        async with self.semaphore:
            try:
                while True:
                    try:
                        response = await self.client.chat.completions.create(**request.body.model_dump(exclude={"type"}))
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


class OllamaAsyncAPI(APIAsync):
    """
    Handles asynchronous requests to the Ollama API.
    """

    def __post_init__(self):
        self.client = AsyncClient(host=self.base_url)
        self.semaphore = asyncio.Semaphore(self.concurrency)

    async def process_single_request(self, request: APIRequest) -> APIOutput:
        async with self.semaphore:
            try:
                response = await self.client.chat(**request.body.model_dump(exclude={"type"}))

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
