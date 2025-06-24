from abc import abstractmethod, ABC
from typing import Generator, Optional

import jinja2
from classconfig import ConfigurableValue, ConfigurableMixin, ConfigurableSubclassFactory

from aicaller.api.base import APIRequest
from aicaller.api.base import OpenAIAPIRequestBody, OllamaAPIRequestBody
from aicaller.loader import Loader
from aicaller.sample_assembler import APISampleAssembler


class Convertor(ConfigurableMixin, ABC):
    """
    Base class for conversion of data.
    """
    loader: Loader = ConfigurableSubclassFactory(Loader, "Loader for the data.")

    @abstractmethod
    def convert(self, p: Optional[str] = None) -> Generator[str, None, None]:
        """
        Converts data.

        :param p: Path to data. If not provided, the path from the configuration is used.
        :return: API request lines
        """
        ...


class ToOpenAIBatchFile(Convertor):
    """
    Base class for conversion of data to OpenAI batch file.
    """

    id_format: str = ConfigurableValue(
        "Format string for custom id. You can use fields {{index}} and fields provided by the sample assembler.",
        user_default="request-{{index}}", voluntary=True)
    model: str = ConfigurableValue("OpenAI model name.", user_default="llama3.2:latest")
    temperature: float = ConfigurableValue("Temperature of the model.", user_default=1.0)
    logprobs: bool = ConfigurableValue("Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.",
                                       user_default=False, voluntary=True)
    max_completion_tokens: int = ConfigurableValue("Maximum number of tokens generated.", user_default=1024)
    sample_assembler: APISampleAssembler = ConfigurableSubclassFactory(APISampleAssembler, "Sample assembler for API request.")
    response_format: Optional[dict] = ConfigurableValue("Format of the response.", voluntary=True,
                                                                   user_default=None)

    def __post_init__(self):
        self.jinja = jinja2.Environment()
        self.jinja_id_template = self.jinja.from_string(self.id_format)

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
    sample_assembler: APISampleAssembler = ConfigurableSubclassFactory(APISampleAssembler, "Sample assembler for API request.")
    format: Optional[dict] = ConfigurableValue("Format of the response.", voluntary=True, user_default=None)

    def __post_init__(self):
        self.jinja = jinja2.Environment()
        self.jinja_id_template = self.jinja.from_string(self.id_format)

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
