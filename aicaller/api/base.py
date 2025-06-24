from abc import ABC, abstractmethod
from typing import Optional, Literal, Union

from classconfig import ConfigurableValue, ConfigurableMixin
from classconfig.validators import StringValidator, MinValueIntegerValidator
from pydantic import BaseModel, Field


class APIConfigMixin(ConfigurableMixin):
    """
    To have a consistent API configuration (one configuration file from user perspective),
    we use this mixin to define the API configuration.
    """

    api_key: str = ConfigurableValue(desc="API key.", validator=StringValidator())
    base_url: Optional[str] = ConfigurableValue(desc="Base URL for API.", user_default=None, voluntary=True)
    pool_interval: Optional[int] = ConfigurableValue(
        desc="Interval in seconds for checking the status of the batch request.",
        user_default=300,
        voluntary=True,
        validator=lambda x: x is None or x > 0)
    process_requests_interval: Optional[int] = ConfigurableValue(
        desc="Interval in seconds between sending requests when processed synchronously.",
        user_default=1,
        voluntary=True,
        validator=lambda x: x is None or x >= 0)
    concurrency: int = ConfigurableValue(
        desc="Maximum number of concurrent requests to the API. This is used with async processing.",
        user_default=10, voluntary=True, validator=MinValueIntegerValidator(1)
    )


class APIRequestBody(BaseModel):
    """
    Represents the body of an API request.
    """
    model: str  # Model to use for the request
    messages: list[dict]  # List of messages in the request

    @property
    @abstractmethod
    def structured(self) -> bool:
        """
        Indicates whether the response is expected to be structured.
        """
        ...


class OpenAIAPIRequestBody(APIRequestBody):
    """
    Represents the body of an OpenAI API request.
    """
    type: Literal["openai"] = "openai"  # Type of the API request
    temperature: float  # Temperature for the model
    logprobs: bool  # Whether to return log probabilities
    max_completion_tokens: int  # Maximum number of tokens to generate
    response_format: Optional[dict] = None  # Format of the response, if any

    @property
    def structured(self) -> bool:
        return self.response_format is not None


class OllamaAPIRequestBody(APIRequestBody):
    """
    Represents the body of an OpenAI API request.
    """
    type: Literal["ollama"] = "ollama"  # Type of the API request
    options: dict  # Options for the model, such as temperature, max tokens, etc.
    format: Optional[dict] = None  # Format of the response, if any

    @property
    def structured(self) -> bool:
        return self.format is not None


class APIRequest(BaseModel):
    """
    Represents a request to an API.
    """
    custom_id: str  # Custom ID for the request
    body: Union[OpenAIAPIRequestBody, OllamaAPIRequestBody] = Field(discriminator='type')


class APIResponse(BaseModel, ABC):
    """
    Represents the response from an API call.
    """
    body: dict
    structured: bool

    @abstractmethod
    def get_raw_content(self, choice: Optional[int] = None) -> str:
        """
        Returns the raw content of the response.

        :param choice: Optional index of the choice to return content from.
        """
        ...


class APIResponseOpenAI(APIResponse):
    type: Literal["openai"] = "openai"

    def get_raw_content(self, choice: Optional[int] = None) -> str:
        """
        Returns the raw content of the response.

        :param choice: Optional index of the choice to return content from.
        :return: Raw content of the response.
        """
        if choice is None:
            choice = 0

        return self.body["choices"][choice]["message"]["content"]


class APIResponseOllama(APIResponse):
    type: Literal["ollama"] = "ollama"

    def get_raw_content(self, choice: Optional[int] = None) -> str:
        if choice is not None:
            raise ValueError("Ollama API does not support multiple choices.")

        return self.body["message"]["content"]


class APIOutput(BaseModel):
    """
    Represents the output of an API call.
    """
    custom_id: str
    response: Optional[Union[APIResponseOpenAI, APIResponseOllama]] = Field(None, discriminator='type')
    error: Optional[str] = None


class APIBase(ABC, APIConfigMixin):
    """
    Base class for API implementations.
    """
    ...
