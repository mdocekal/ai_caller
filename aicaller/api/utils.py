from typing import Optional, Sequence

from google import genai
from google.genai.types import Part

from aicaller.api.base import APIRequest


class GoogleGenAIAPIMixin:

    @classmethod
    def convert_part(cls, part: str | Sequence[str]) -> Part:
        """
        Converts a string message or different modalities to a Google GenAI Part.

        :param part: A string message or non text modality represented as a tuple of (mime_type, file_path).
        :return: A genai.types.Part object.
        """
        if isinstance(part, str):
            return Part.from_text(text=part)
        elif isinstance(part, (tuple, list)) and len(part) == 2:
            mime_type, file_path = part
            with open(file_path, "rb") as f:
                image_bytes = f.read()

            return Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            )
        else:
            raise ValueError("Invalid part format. Must be a string or a tuple of (mime_type, file_path).")

    @classmethod
    def get_conversion_history(cls, request: APIRequest) -> list[genai.types.Content]:
        """
        Converts APIRequest messages to Google GenAI conversation history format.

        :param request: APIRequest object.
        :return: List of genai.types.Content representing the conversation history.
        """
        conversation_history = [
            genai.types.Content(
                role="model" if message["role"] == "assistant" else message["role"],
                parts=[cls.convert_part(p) for p in message["parts"]]
            )
            for message in request.body.messages if message["role"] in {"user", "assistant", "model"}
        ]
        return conversation_history

    @classmethod
    def get_config(cls, request: APIRequest) -> Optional[genai.types.GenerateContentConfig]:
        """
        Converts APIRequest body config to Google GenAI GenerateContentConfig.

        :param request: APIRequest object.
        :return: genai.types.GenerateContentConfig object.
        """
        args = {}

        # get system message
        for message in request.body.messages:
            if message["role"] == "system":
                if len(message["parts"]) > 1:
                    raise ValueError("Google GenAI API supports only single-part system messages.")
                args["system_instruction"] = genai.types.Content(
                    parts=[genai.types.Part(text=message["parts"][0])]
                )
                break

        for key, value in request.body.options.items():
            args[key] = value

        if request.body.format is not None:
            args["response_mime_type"] = "application/json"
            args["response_json_schema"] = request.body.format if isinstance(request.body.format, dict) else request.body.format.model_json_schema()

        if len(args) == 0:
            return None

        return genai.types.GenerateContentConfig(
            **args
        )
