# Paraphrasing example

In this example we will create a batch file from existing dataset available on Hugging Face hub. For the sake of simplicity we will use the Microsoft Research Paraphrase Corpus (MRPC) dataset.
This dataset is available as a part of glue benchmark. It contains pairs of sentences and a label indicating whether the sentences are paraphrases or not.

We are going to make an experiment whether it is better to use simple one turn prompt or a request that resembles a chat history.

## One turn prompt
Firstly we need to convert the dataset to a batch file. For that we need to create a configuration file:
```bash
aicaller create_config --path config_simple.yaml
```

Using the guide we will select the following options:
* create batch workflow
* ToOpenAIBatchFile
* HFLoader
* TextDatasetAssembler
* StringTemplate

This will create a configuration file, and we will edit it to look like this:
```yaml
convertor:  # Convertor to batch file.
  cls: ToOpenAIBatchFile  # name of class that is subclass of Convertor
  config: # configuration for defined class
    loader:  # Loader for the data.
      cls: HFLoader  # name of class that is subclass of Loader
      config: # configuration for defined class
        path_to: nyu-mll/glue # Path to the data.
        config: mrpc # Configuration name.
        split: test # Split of the dataset.
    id_format: "{{idx}}" # Format string for custom id. You can use fields {{index}} and fields provided by the sample assembler.
    model: gpt-4o-mini # OpenAI model name.
    temperature: 1.0 # Temperature of the model.
    logprobs: false # Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.
    max_completion_tokens: 1024 # Maximum number of tokens generated.
    sample_assembler: # Sample assembler for API request.
      cls: TextDatasetAssembler  # name of class that is subclass of APISampleAssembler
      config: # configuration for defined class
        few_shot_sampler:  # Few shot sampler for sampling examples. It will save few-shot indices and samples to few_shot_indices and few_shot fields.
          load:  # Loader for the data.
            cls: HFLoader  # name of class that is subclass of Loader
            config: # configuration for defined class
              path_to: nyu-mll/glue # Path to the data.
              config: mrpc # Configuration name.
              split: train # Split of the dataset.
          n: 3 # Number of examples to sample.
        id_fields: # List of additional dataset fields to use for sample ids.
          - idx
        input_template: # Template for input assembly.
          cls: StringTemplate  # name of class that is subclass of Template
          config: # configuration for defined class
            template: | # Jinja2 template for prompt sequence.
              Decide whether the sentences are paraphrases or not. Answer with 1 for yes and 0 for no.
              {{few_shot[0]["sentence1"]}}
              {{few_shot[0]["sentence2"]}}
              {"label":{{few_shot[0]["label"]}}}
              {{few_shot[1]["sentence1"]}}
              {{few_shot[1]["sentence2"]}}
              {"label":{{few_shot[1]["label"]}}}
              {{few_shot[2]["sentence1"]}}
              {{few_shot[2]["sentence2"]}}
              {"label":{{few_shot[2]["label"]}}}
              {{sentence1}}
              {{sentence2}}
        direct: # Name of jsonl field that contains the sample. In that case, the template is not used.
    response_format: # Format of the response.
        type: json_schema
        json_schema:
          name: paraphrase
          strict: true
          schema:
            type: object
            properties:
              label:
                type: integer
            additionalProperties: false
            required:
                - label
```

Let's break down the configuration file:

* `convertor` - defines the convertor to be used. In this case we are using `ToOpenAIBatchFile` convertor, as we want to use OpenAI API.
* `loader` - we specify the dataset which we want to use. In this case it is the MRPC test set that is part of the glue benchmark.
* `id_format` - format string for custom id. You can use fields {{index}} and fields provided by the sample assembler, that are defined in the `id_fields` attribute.
* `model` - name of the model to be used. In this case we are using `gpt-4o-mini` model.
* `sample_assembler` - defines the sample assembler to be used. In this case we are using `TextDatasetAssembler` as we are working with text data.
  * `few_shot_sampler` - In this example we want to use few-shot learning, so we randomly sample 3 examples from the training set. Those samples are then available in the template as `few_shot` variable. If needed, you can also use `few_shot_indices` variable to get the indices of the sampled examples.
  * `id_fields` - list of additional dataset fields to use for sample ids. In this case we are using `idx` field from the dataset.
  * `input_template` - in this case we are using simple string template, that is used to create the input for the model. It uses Jinja2 template syntax. It will be transformed to one turn conversation with just user turn.
* `response_format` - defines the format of the response. In this case we are using `json_schema` format, which is a JSON schema that defines the expected output. It is used to validate the response from the API. In this case we expect a JSON object with a single field `label` that is an integer.

Now we can create the batch file using the following command:
```bash
aicaller create_batch_file --config config_simple.yaml > batch_simple.jsonl
```

This will create a batch file with the requests to the API. You can check the batch file to see the requests that will be sent to the API.

Before we send the requests to the API, we need to create a configuration file for the API. For that we can use the `create_config` command again:
```bash
aicaller create_config --path api_config.yaml
```

Using the guide we will select the following options:
* API
* OpenAPIFactory

This will create a configuration file like this:
```yaml

api:  # API type
  cls: OpenAPIFactory  # name of class that is subclass of APIFactory
  config: # configuration for defined class
    api_key:  # API key.
    base_url: # Base URL for API.
    id_field: custom_id # Field name that contains the request ID.
    pool_interval: 300 # Interval in seconds for checking the status of the batch request.
    process_requests_interval: 1 # Interval in seconds between sending requests when processed synchronously.
    concurrency: 10 # Maximum number of concurrent requests to the API. This is used with async processing.

```

After filling in the API key, we can send the requests to the API using the following command:
```bash
aicaller batch_request batch_simple.jsonl -c api_config.yaml -r results.jsonl
```
This will send batch request to the API and save the results to the `results.jsonl` file. You can check the results file to see the responses from the API.

## Chat
In this case we will use the same dataset, but we will use a chat history as a template. Again use the `create_config` command to create a configuration file:
```bash
aicaller create_config --path config_chat.yaml
```

Using the guide we will select the following options:
* create batch workflow
* ToOpenAIBatchFile
* HFLoader
* TextDatasetAssembler
* MessagesTemplate


```yaml
convertor:  # Convertor to batch file.
  cls: ToOpenAIBatchFile  # name of class that is subclass of Convertor
  config: # configuration for defined class
    loader:  # Loader for the data.
      cls: HFLoader  # name of class that is subclass of Loader
      config: # configuration for defined class
        path_to: nyu-mll/glue # Path to the data.
        config: mrpc # Configuration name.
        split: test # Split of the dataset.
    id_format: "{{idx}}" # Format string for custom id. You can use fields {{index}} and fields provided by the sample assembler.
    model: gpt-4o-mini # OpenAI model name.
    temperature: 1.0 # Temperature of the model.
    logprobs: false # Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.
    max_completion_tokens: 1024 # Maximum number of tokens generated.
    sample_assembler: # Sample assembler for API request.
      cls: TextDatasetAssembler  # name of class that is subclass of APISampleAssembler
      config: # configuration for defined class
        few_shot_sampler:  # Few shot sampler for sampling examples. It will save few-shot indices and samples to few_shot_indices and few_shot fields.
          load:  # Loader for the data.
            cls: HFLoader  # name of class that is subclass of Loader
            config: # configuration for defined class
              path_to: nyu-mll/glue # Path to the data.
              config: mrpc # Configuration name.
              split: train # Split of the dataset.
          n: 3 # Number of examples to sample.
        id_fields: # List of additional dataset fields to use for sample ids.
          - idx
        input_template: # Template for input assembly.
          cls: MessagesTemplate  # name of class that is subclass of Template
          config: # configuration for defined class
            messages:  # List of message builders.
                - cls: OpenAIMessageBuilder
                  config:
                    role: system
                    content: | # Jinja2 template for prompt sequence.
                      You are a helpful assistant that decides whether the sentences are paraphrases or not. Answer with 1 for yes and 0 for no using json format with a single field "label".
                -
                  cls: OpenAIMessageBuilder
                  config:
                    role: user
                    content: | # Jinja2 template for prompt sequence.
                      Decide whether given sentences are paraphrases or not.
                      {{few_shot[0]["sentence1"]}}
                      {{few_shot[0]["sentence2"]}}
                -
                    cls: OpenAIMessageBuilder
                    config:
                        role: assistant
                        content: | # Jinja2 template for prompt sequence.
                          {"label":{{few_shot[0]["label"]}}}
                -
                    cls: OpenAIMessageBuilder
                    config:
                        role: user
                        content: | # Jinja2 template for prompt sequence.
                          {{few_shot[1]["sentence1"]}}
                          {{few_shot[1]["sentence2"]}}
                -
                    cls: OpenAIMessageBuilder
                    config:
                        role: assistant
                        content: | # Jinja2 template for prompt sequence.
                          {"label":{{few_shot[1]["label"]}}}
                -
                    cls: OpenAIMessageBuilder
                    config:
                        role: user
                        content: | # Jinja2 template for prompt sequence.
                          {{few_shot[2]["sentence1"]}}
                          {{few_shot[2]["sentence2"]}}
                -
                    cls: OpenAIMessageBuilder
                    config:
                        role: assistant
                        content: | # Jinja2 template for prompt sequence.
                          {"label":{{few_shot[2]["label"]}}}
                -
                    cls: OpenAIMessageBuilder
                    config:
                        role: user
                        content: | # Jinja2 template for prompt sequence.
                          {{sentence1}}
                          {{sentence2}}
        direct: # Name of jsonl field that contains the sample. In that case, the template is not used.
    response_format: # Format of the response.
        type: json_schema
        json_schema:
          name: paraphrase
          strict: true
          schema:
            type: object
            properties:
              label:
                type: integer
            additionalProperties: false
            required:
                - label
```

This configuration file is similar to the previous one, but we are using `MessagesTemplate` with OpenAIMessageBuilder.

Right now there are two message builders available:
* `OpenAIMessageBuilder` - for OpenAI compatible APIs
* `OpenAIMultiModalMessageBuilder` - for OpenAI compatible APIs with multimodal support
* `OllamaMessageBuilder` - for Ollama API, this message builder supports text and image messages.

As before, we can create the batch file using the following command:
```bash
aicaller create_batch_file --config config_chat.yaml > batch_chat.jsonl
```

After that we can use the same API configuration file as before and send the requests to the API using the following command:
```bash
aicaller batch_request batch_chat.jsonl -c api_config.yaml -r results.jsonl
```
