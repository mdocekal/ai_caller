import json
from typing import Generator, Optional

import jinja2
from classconfig import ConfigurableValue, ConfigurableMixin, ConfigurableSubclassFactory

from openaiapicaller.sample_assembler import APISampleAssembler


class ToOpenAIBatchFile(ConfigurableMixin):
    """
    Base class for conversion of data to OpenAI batch file.
    """

    id_format: str = ConfigurableValue(
        "Format string for custom id. You can use fields {{index}} and fields provided by the sample assembler.",
        user_default="request-{{index}}", voluntary=True)
    model: str = ConfigurableValue("OpenAI model name.", user_default="gpt-4o-mini")
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

    def convert(self, p: str) -> Generator[str, None, None]:
        """
        Converts IR annotations to OpenAI batch file.

        :param p: Path to data
        :return: OpenAI batch file lines
        """
        for i, (sample, sample_ids) in enumerate(self.sample_assembler.assemble(p)):
            custom_id_fields = {**sample_ids, "index": i}
            request = {
                "custom_id": self.jinja_id_template.render(custom_id_fields),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": sample}
                    ],
                    "temperature": self.temperature,
                    "logprobs": self.logprobs,
                    "max_completion_tokens": self.max_completion_tokens
                }
            }
            if self.response_format is not None:
                request["body"]["response_format"] = self.response_format

            yield json.dumps(request, ensure_ascii=False)
