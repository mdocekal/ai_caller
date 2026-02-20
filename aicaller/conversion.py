import importlib
from abc import abstractmethod, ABC
from typing import Generator, Optional, Type

import jinja2
from classconfig import ConfigurableValue, ConfigurableMixin, ConfigurableSubclassFactory, RelativePathTransformer
from pydantic import BaseModel

from aicaller.api.base import APIRequest, GoogleGenAIAPIRequestBody
from aicaller.api.base import OpenAIAPIRequestBody, OllamaAPIRequestBody
from aicaller.loader import Loader
from aicaller.sample_assembler import APISampleAssembler
from aicaller.modules import load_module


class Convertor(ConfigurableMixin, ABC):
    """
    Base class for conversion of data.
    """
    loader: Loader = ConfigurableSubclassFactory(Loader, "Loader for the data.")
    method: str = ConfigurableValue("HTTP method for the request", user_default="POST", voluntary=True, validator=lambda x: x in ["POST"])
    url: str = ConfigurableValue("URL for the request", user_default="/v1/chat/completions", voluntary=True)

    @abstractmethod
    def convert(self, p: Optional[str] = None) -> Generator[str, None, None]:
        """
        Converts data.

        :param p: Path to data. If not provided, the path from the configuration is used.
        :return: API request lines
        """
        ...

    @staticmethod
    def load_response_class(path_to_module: str) -> Type[BaseModel]:
        """
        Loads a response class from a module.

        :param path_to_module: Path to the module.
        :return: Loaded response class.
        :raises ValueError: If the response class is not a pydantic BaseModel.
        """
        module = load_module(path_to_module)
        response_class = getattr(module, "RESPONSE_FORMAT")
        if not issubclass(response_class, BaseModel):
            raise ValueError(f"Response class must be a pydantic BaseModel, got {type(response_class)}")
        return response_class


class ToOpenAIBatchFile(Convertor):
    """
    Base class for conversion of data to OpenAI batch file.
    """

    id_format: str = ConfigurableValue(
        "Format string for custom id. You can use fields {{index}} and fields provided by the sample assembler.",
        user_default="request-{{index}}", voluntary=True)
    model: str = ConfigurableValue("OpenAI model name.", user_default="llama3.2:latest")
    temperature: float = ConfigurableValue("Temperature of the model.", user_default=1.0)
    logprobs: bool = ConfigurableValue(
        "Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.",
        user_default=False, voluntary=True)
    max_completion_tokens: int = ConfigurableValue("Maximum number of tokens generated.", user_default=1024)
    sample_assembler: APISampleAssembler = ConfigurableSubclassFactory(APISampleAssembler,
                                                                       "Sample assembler for API request.")
    response_format: Optional[dict | str | Type[BaseModel]] = ConfigurableValue("Format of the response.",
                                                                                voluntary=True,
                                                                                user_default=None)
    response_format_path: Optional[str] = ConfigurableValue(
        "If set, the format will be loaded from this path. It must point to a python file with a RESPONSE_FORMAT variable containing subclass of pydantic BaseModel. This has precedence over format.",
        voluntary=True,
        user_default=None,
        transform=RelativePathTransformer(allow_none=True)
    )

    def __post_init__(self):
        self.jinja = jinja2.Environment()
        self.jinja_id_template = self.jinja.from_string(self.id_format)

        if isinstance(self.response_format_path, str):
            resp = self.load_response_class(self.response_format_path)
            self.response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": resp.__name__,
                    "schema": resp.model_json_schema()
                }
            }

    def build_request(self, sample: list[dict], custom_id_fields: dict) -> APIRequest:
        """
        Builds a request for the API.

        :param sample: Sample with messages.
        :param custom_id_fields: Fields for custom id.
        :return: Request for the API.
        """

        if isinstance(sample, str):
            sample = [{"role": "user", "content": sample}]

        body = OpenAIAPIRequestBody(
            model=self.model,
            messages=sample,
            temperature=self.temperature,
            logprobs=self.logprobs,
            max_completion_tokens=self.max_completion_tokens,
            response_format=self.response_format
        )
        request = APIRequest(
            custom_id=self.jinja_id_template.render(custom_id_fields),
            body=body
        )
        return request

    def convert(self, p: Optional[str] = None) -> Generator[str, None, None]:
        """
        Converts to OpenAI batch file.

        :param p: Path to data
        :return: OpenAI batch file lines
        """
        dataset = self.loader.load(p)
        for i, (sample, sample_ids) in enumerate(self.sample_assembler.assemble(dataset)):
            request = self.build_request(
                sample=sample,
                custom_id_fields={**sample_ids, "index": i}
            )
            yield request.model_dump_json()


class ToOllamaBatchFile(Convertor):
    """
    Base class for conversion of data to Ollama batch file.
    """

    id_format: str = ConfigurableValue(
        "Format string for custom id. You can use fields {{index}} and fields provided by the sample assembler.",
        user_default="request-{{index}}", voluntary=True)
    model: str = ConfigurableValue("model name", user_default="llama3.2:latest")
    options: dict = ConfigurableValue(
        "additional model parameters listed in the documentation for the Modelfile such as temperature",
        user_default={"temperature": 1.0, "num_ctx": 2048, "num_predict": 128}
    )
    sample_assembler: APISampleAssembler = ConfigurableSubclassFactory(APISampleAssembler,
                                                                       "Sample assembler for API request.")
    format: Optional[dict | str | Type[BaseModel]] = ConfigurableValue("Format of the response.",
                                                                       voluntary=True,
                                                                       user_default=None)
    format_path: Optional[str] = ConfigurableValue(
        "If set, the format will be loaded from this path. It must point to a python file with a RESPONSE_FORMAT variable containing subclass of pydantic BaseModel. This has precedence over format.",
        voluntary=True,
        user_default=None,
        transform=RelativePathTransformer(allow_none=True)
    )

    def __post_init__(self):
        self.jinja = jinja2.Environment()
        self.jinja_id_template = self.jinja.from_string(self.id_format)
        if isinstance(self.format_path, str):
            self.format = self.load_response_class(self.format_path).model_json_schema()

    def build_request(self, sample: list[dict], custom_id_fields: dict) -> APIRequest:
        """
        Builds a request for the API.

        :param sample: Sample with messages.
        :param custom_id_fields: Fields for custom id.
        :return: Request for the API.
        """

        if isinstance(sample, str):
            sample = [{"role": "user", "content": sample}]

        body = OllamaAPIRequestBody(
            model=self.model,
            messages=sample,
            options=self.options,
            format=self.format
        )
        request = APIRequest(
            custom_id=self.jinja_id_template.render(custom_id_fields),
            body=body
        )

        return request

    def convert(self, p: Optional[str] = None) -> Generator[str, None, None]:
        """
        Converts IR annotations to OpenAI batch file.

        :param p: Path to data
        :return: OpenAI batch file lines
        """
        dataset = self.loader.load(p)

        for i, (sample, sample_ids) in enumerate(self.sample_assembler.assemble(dataset)):
            request = self.build_request(
                sample=sample,
                custom_id_fields={**sample_ids, "index": i}
            )
            yield request.model_dump_json()


class ToGoogleGenAIBatchFile(Convertor):
    """
    Base class for conversion of data to GoogleGenAI batch file.
    """

    id_format: str = ConfigurableValue(
        "Format string for custom id. You can use fields {{index}} and fields provided by the sample assembler.",
        user_default="request-{{index}}", voluntary=True)
    model: str = ConfigurableValue("model name", user_default="llama3.2:latest")
    options: dict = ConfigurableValue(
        "additional model parameters, see official google documentation for available options",
        user_default={"max_output_tokens": 2048}
    )
    sample_assembler: APISampleAssembler = ConfigurableSubclassFactory(APISampleAssembler,
                                                                       "Sample assembler for API request.")
    format: Optional[dict | str | Type[BaseModel]] = ConfigurableValue("Format of the response.",
                                                                       voluntary=True,
                                                                       user_default=None)
    format_path: Optional[str] = ConfigurableValue(
        "If set, the format will be loaded from this path. It must point to a python file with a RESPONSE_FORMAT variable containing subclass of pydantic BaseModel. This has precedence over format.",
        voluntary=True,
        user_default=None,
        transform=RelativePathTransformer(allow_none=True)
    )

    def __post_init__(self):
        self.jinja = jinja2.Environment()
        self.jinja_id_template = self.jinja.from_string(self.id_format)
        if isinstance(self.format_path, str):
            self.format = self.load_response_class(self.format_path).model_json_schema()

    def build_request(self, sample: list[dict], custom_id_fields: dict) -> APIRequest:
        """
        Builds a request for the API.

        :param sample: Sample with messages.
        :param custom_id_fields: Fields for custom id.
        :return: Request for the API.
        """

        if isinstance(sample, str):
            sample = [{"role": "user", "content": sample}]

        body = GoogleGenAIAPIRequestBody(
            model=self.model,
            messages=sample,
            options=self.options,
            format=self.format
        )
        request = APIRequest(
            custom_id=self.jinja_id_template.render(custom_id_fields),
            body=body
        )

        return request

    def convert(self, p: Optional[str] = None) -> Generator[str, None, None]:
        """
        Converts IR annotations to OpenAI batch file.

        :param p: Path to data
        :return: OpenAI batch file lines
        """
        dataset = self.loader.load(p)

        for i, (sample, sample_ids) in enumerate(self.sample_assembler.assemble(dataset)):
            request = self.build_request(
                sample=sample,
                custom_id_fields={**sample_ids, "index": i}
            )
            yield request.model_dump_json()
