import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

import jinja2
from classconfig import ConfigurableValue, ListOfConfigurableSubclassFactoryAttributes, ConfigurableSubclassFactory
from classconfig.transforms import EnumTransformer
from classconfig.validators import ListOfTypesValidator, StringValidator, AnyValidator, IsNoneValidator
from segmentedstring import SegmentedString

from aicaller.utils import is_url, obtain_base64_image, detect_image_format

from google.genai import types as genai_types


class Jinja2EnvironmentSingletonFactory:
    """
    Singleton factory for Jinja2 environment.
    It is here to be able to add additional filters.
    """
    jinja_env: jinja2.Environment = None

    def __init__(self):
        if not self.jinja_env:
            self.jinja_env = jinja2.Environment()
            # default tojson doesn't allow to use all the arguments of json.dumps
            self.jinja_env.filters["tojson"] = json.dumps
            self.jinja_env.filters["fromjson"] = json.loads
            self.jinja_env.filters["filter_dict"] = lambda d, keys: {k: v for k, v in d.items() if k in keys}
            self.jinja_env.filters["model_dump_json"] = lambda obj: obj.model_dump_json()


class Template(ABC):
    """
    Abstract base class for all prompt templates.
    """

    @abstractmethod
    def render(self, data: dict[str, Any]) -> Any:
        """
        Renders the template with the data.

        :param data: data
        :return: rendered template
        """
        ...


class StringTemplate(Template):
    """
    Template for simple string prompt.
    """

    template: str = ConfigurableValue("Jinja2 template for prompt sequence.")

    def __init__(self, template: str):
        self.template = template
        self.jinja = Jinja2EnvironmentSingletonFactory().jinja_env
        self.jinja_template = self.jinja.from_string(template)

    def render(self, data: dict[str, Any]) -> str:
        return self.jinja_template.render(data)


class SegmentedStringTemplate(Template):
    """
    Template for segmented string prompt.
    """

    template: dict[str, str] = ConfigurableValue(
        "Dictionary with keys 'segment_name' and 'template' (the template is jinja2 template). All parts will be concatenated and SegmentedString will be used for the rendered result."
    )

    def __init__(self, template: dict[str, str]):
        self.template = template
        self.jinja = Jinja2EnvironmentSingletonFactory().jinja_env
        self.jinja_template = {
            k: self.jinja.from_string(v) for k, v in template.items()
        }

    def render(self, data: dict[str, Any]) -> SegmentedString:
        return SegmentedString(
            [self.jinja_template[k].render(data) for k in self.jinja_template.keys()],
            self.jinja_template.keys()
        )


class MessageBuilder(ABC):
    """
    ABC for chat message builders.
    """

    @abstractmethod
    def render(self, data: dict[str, Any]) -> Any:
        """
        Renders the message with the data.

        :param data: data
        :return: rendered message
        """
        ...


class OpenAIMessageBuilder(MessageBuilder):
    """
    Builder for OpenAI chat message.
    """

    role: str = ConfigurableValue("Role of the message. Such as 'system', 'user', 'assistant'.",
                                  validator=StringValidator())
    content: str = ConfigurableValue("Jinja2 template for text content of the message.", validator=StringValidator())

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        self.jinja = Jinja2EnvironmentSingletonFactory().jinja_env
        self.jinja_template = self.jinja.from_string(content)

    def render(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.jinja_template.render(data)
        }


class OpenAIContentType(ABC):
    """
    ABC for OpenAI content types.
    """

    @abstractmethod
    def render(self, data: dict[str, Any]) -> Any:
        """
        Renders the content with the data.

        :param data: data
        :return: rendered content
        """
        ...


class OpenAITextContent(OpenAIContentType):
    """
    Text content for OpenAI message.
    """

    text: str = ConfigurableValue("Jinja2 template for text content of the message.", validator=StringValidator())

    def __init__(self, text: str):
        self.text = text
        self.jinja = Jinja2EnvironmentSingletonFactory().jinja_env
        self.jinja_template = self.jinja.from_string(text)

    def render(self, data: dict[str, Any]) -> dict[str, str]:
        return {
            "type": "text",
            "text": self.jinja_template.render(data)
        }


class OpenAIImageDetail(Enum):
    """
    Enum for image detail.
    """

    AUTO = "auto"
    LOW = "low"
    HIGH = "high"


class OpenAIImageContent(OpenAIContentType):
    """
    Image content for OpenAI message.
    """

    url: str = ConfigurableValue("Simple jinja template containing url or path to the image.", validator=StringValidator())
    detail: OpenAIImageDetail = ConfigurableValue(
        "Detail of the image. Can be 'auto', 'low' or 'high'.",
        user_default=OpenAIImageDetail.AUTO,
        transform=EnumTransformer(OpenAIImageDetail)
    )

    def __init__(self, url: str, detail: OpenAIImageDetail = OpenAIImageDetail.AUTO):
        self.url = url
        self.detail = detail
        self.jinja = Jinja2EnvironmentSingletonFactory().jinja_env
        self.jinja_template = self.jinja.from_string(url)

    def render(self, data: dict[str, Any]) -> dict[str, str]:
        image_path = self.jinja_template.render(data)
        if is_url(image_path):
            return {
                "type": "image_url",
                "image_url": {
                    "url": image_path,
                    "detail": self.detail.value
                }
            }
        else:
            # get image format
            image_format = detect_image_format(image_path)
            try:
                mime_type = {
                    "png": "image/png",
                    "jpeg": "image/jpeg",
                    "webp": "image/webp",
                    "gif": "image/gif"
                }[image_format]
            except KeyError:
                raise ValueError(f"Unsupported image format: {image_format}")

            base64_image = obtain_base64_image(image_path)

            return {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:{mime_type};base64,{base64_image}",
                    "detail": self.detail.value
                }
            }


class OpenAIMultiModalMessageBuilder(MessageBuilder):
    """
    Builder for OpenAI chat message.
    """

    role: str = ConfigurableValue("Role of the message. Such as 'system', 'user', 'assistant'.",
                                  validator=StringValidator())
    content: list[OpenAIContentType] = ListOfConfigurableSubclassFactoryAttributes(
        ConfigurableSubclassFactory(OpenAIContentType, "Content of the message."),
        "Content of the message."
    )

    def __init__(self, role: str, content: list[OpenAIContentType]):
        self.role = role
        self.content = content

    def render(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": [content.render(data) for content in self.content]
        }


class OllamaMessageBuilder(MessageBuilder):
    """
    Builder for Ollama chat message.
    """

    role: str = ConfigurableValue("Role of the message. Such as 'system', 'user', 'assistant'.",
                                  validator=StringValidator())
    content: str = ConfigurableValue("Jinja2 template for text content of the message.", validator=StringValidator())
    images: Optional[list[str]] = ConfigurableValue("Jinja2 templates for paths to images.",
                                                    voluntary=True,
                                                    validator=AnyValidator([IsNoneValidator(), ListOfTypesValidator(str, allow_empty=True)]))

    def __init__(self, role: str, content: str, images: Optional[list[str]] = None):
        self.role = role
        self.content = content
        self.images = images

        self.jinja = Jinja2EnvironmentSingletonFactory().jinja_env
        self.jinja_template = self.jinja.from_string(content)
        if images:
            self.jinja_images = [self.jinja.from_string(image) for image in images]

    def render(self, data: dict[str, Any]) -> dict[str, Any]:
        message = {
            "role": self.role,
            "content": self.jinja_template.render(data)
        }
        if self.images:
            message["images"] = [jinja_image.render(data) for jinja_image in self.jinja_images]
        return message


class GoogleGenAIMessageBuilder(MessageBuilder):
    """
    Builder for GoogleGenAI chat message.

    GoogleGenAI uses a 'parts' array format where each part can be text or an image.
    Images are represented as PIL Image objects or base64-encoded data.
    """

    role: str = ConfigurableValue("Role of the message. Such as 'system', 'user', 'model' (not 'assistant').",
                                  validator=StringValidator())
    content: str = ConfigurableValue("Jinja2 template for text content of the message.", validator=StringValidator())
    images: Optional[list[str]] = ConfigurableValue("Jinja2 templates for paths to images.",
                                                    voluntary=True,
                                                    validator=AnyValidator([IsNoneValidator(), ListOfTypesValidator(str, allow_empty=True)]))

    def __init__(self, role: str, content: str, images: Optional[list[str]] = None):
        self.role = role
        self.content = content
        self.images = images

        self.jinja = Jinja2EnvironmentSingletonFactory().jinja_env
        self.jinja_template = self.jinja.from_string(content)
        if images:
            self.jinja_images = [self.jinja.from_string(image) for image in images]

    def render(self, data: dict[str, Any]) -> dict[str, Any]:
        # GoogleGenAI expects 'parts' array with text and optional images
        parts = []

        # Add text content
        text_content = self.jinja_template.render(data)
        if text_content:
            parts.append(text_content)

        # Add images if present
        if self.images:
            for jinja_image in self.jinja_images:
                image_path = jinja_image.render(data)

                image_format = detect_image_format(image_path)
                mime_type = {
                    "png": "image/png",
                    "jpeg": "image/jpeg",
                    "jpg": "image/jpeg",
                    "webp": "image/webp",
                    "gif": "image/gif"
                }.get(image_format, f"image/{image_format}")

                parts.append((mime_type, image_path))

        message = {
            "role": self.role,
            "parts": parts
        }

        return message


class MessagesTemplate(Template):
    """
    Template for chat messages prompt.
    """

    messages: list[MessageBuilder] = ListOfConfigurableSubclassFactoryAttributes(
        ConfigurableSubclassFactory(MessageBuilder, "Message builder."), "List of message builders."

    )

    def __init__(self, messages: list[MessageBuilder]):
        self.messages = messages

    def render(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        return [message.render(data) for message in self.messages]

