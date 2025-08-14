# Image Captioning Example

This example demonstrates how to process images and generate captions. We will use toy data stored in `data` folder in this example directory.

## TL;DR
Here is a quick overview of the steps to follow:
1. Use the provided config (`config_openai.yaml` or `config_ollama.yaml`) for creating a batch of API requests or create your own configuration file (see [Usage](#usage) section).
2. Create a batch of requests using this command:
   ```bash
   aicaller create_batch_file --config config_openai.yaml > batch_openai.jsonl
   ```
3. Create (see [Sending the batch file](#sending-the-batch-file) section) or edit the API configuration file (`api.yaml`) to include your API key and other settings.
4. Send the batch file to the API using this command:
   ```bash
   aicaller batch_request batch_openai.jsonl --config api.yaml --synchronous --results results_openai/ --only_output
   ```
   You can also use `--asynchronous` flag to send requests in parallel or omit the argument entirely to send requests in batch mode (supported by OpenAI, see more at [https://platform.openai.com/docs/guides/batch](https://platform.openai.com/docs/guides/batch)).


## Data
The `data` folder contains images that will be used for captioning. It also contains a `metadata.jsonl` file that provides metadata for each image. Each metadata record contains file name and additional information such as the title of the document from which the image comes from.

```json lines
{"file_name": "uuid_3A0ab2a2df-4d9b-4330-b809-2790138a6599__obrázek_0.jpg", "doc_title": "Architectural Sketch – Residential Design Draft", "caption": "A hand-drawn architectural concept for a single-family home.", "author": "Ing. Petra Nováková", "created_at": "2023-03-14T10:22:00Z", "document_type": "architectural_drawing", "tags": ["architecture", "sketch", "residential", "draft"]}
{"file_name": "uuid_3A0a6c6a12-3dd8-450e-a2ef-6bbc36996c04__obrázek_1.jpg", "doc_title": "Historical Building Facade – Reference Image", "caption": "Photograph of a restored 19th-century facade used for conservation planning.", "author": "Bc. Karel Dvořák", "created_at": "2022-09-08T08:10:45Z", "document_type": "photograph", "tags": ["heritage", "architecture", "facade", "restoration"]}
```
## Usage
Firstly, we need to create a configuration file to create a batch of requests. We can then send those requests to the given API.

### OpenAI configuration
To create brand new configuration file run the following command:
```bash
aicaller create_config --path config_openai.yaml
```
Using the guide, we select the following options:
* create batch workflow
* ToOpenAIBatchFile
* HFImageLoader
* ImageDatasetAssembler
* MessagesTemplate

This will create a configuration file, and we will edit it to look like this:
```yaml
convertor:  # Convertor to batch file.
  cls: ToOpenAIBatchFile  # name of class that is subclass of Convertor
  config: # configuration for defined class
    loader:  # Loader for the data.
      cls: HFImageLoader  # name of class that is subclass of Loader
      config: # configuration for defined class
        path_to: data # Path to the data.
        config: # Configuration name.
        split: train # Split of the dataset.
    id_format: "{{file_name}}" # Format string for custom id. You can use fields {{index}} and fields provided by the sample assembler.
    model: gpt-4o-mini # OpenAI model name.
    temperature: 1.0 # Temperature of the model.
    logprobs: false # Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.
    max_completion_tokens: 1024 # Maximum number of tokens generated.
    sample_assembler: # Sample assembler for API request.
      cls: ImageDatasetAssembler  # name of class that is subclass of APISampleAssembler
      config: # configuration for defined class
        input_template: # Template for input assembly.
          cls: MessagesTemplate  # name of class that is subclass of Template
          config: # configuration for defined class
            messages:   # List of message builders.
              - cls: OpenAIMultiModalMessageBuilder
                config:
                  role: user
                  content:
                    - cls: OpenAITextContent  # name of class that is subclass of Content
                      config: # configuration for defined class
                        text: |
                          Describe the image in a few sentences. Describe the image itself (photo, drawing, graphs), possibly its style and historic period. Describe the image content, objects, actions, location, names.
                        
                          Use the following metadata to help you:
                            document title: {{ doc_title }}
                            document author: {{ author }}
                            tags: {{ tags | join(", ") }}
                    - cls: OpenAIImageContent  # name of class that is subclass of Content
                      config: # configuration for defined class
                        url: "{{image.filename}}"  # Text of the message.
                        detail: LOW
    response_format: # Format of the response.
```

Let's break down the configuration file:

* `convertor` - defines the convertor to be used. In this case we are using `ToOpenAIBatchFile` convertor, as we want to use OpenAI API.
* `loader` - we specify the dataset which we want to use. In this case it is the toy image dataset stored in `data` folder. Thus, we use `HFImageLoader` as we use the Hugging Face Image Dataset format.
  * `path_to` - path to the dataset. In this case it is `data` folder.
  * `config` - configuration name. In this case we are using default configuration.
  * `split` - split of the dataset to use. In this case we are using `train` split which is the default split that is created when loading dataset from the `data` folder.
* `id_format` - format string for custom id. You can use fields {{index}}, {{file_name}}, and {{image_path}}.
* `model` - name of the model to be used. In this case we are using `gpt-4o-mini` model.
* `sample_assembler` - defines the sample assembler to be used. In this case we are using `ImageDatasetAssembler` as we are working with image dataset.
  * `input_template` - We are providing a chat style template with text prompt and image for which we want to generate a caption. You can see that the prompt is jinja2 template in which we are able to use the metadata fields directly as they are automatically provided.
* `response_format` - this could be used to define the output format of the response. In this case we are not using it as we want to get a plain text response.

Now we can create the batch file using the following command:
```bash
aicaller create_batch_file --config config_openai.yaml > batch_openai.jsonl
```

This will create a batch file with the requests to the API. You can check the batch file to see the requests that will be sent to the API.

It is also possible to separate the configuration for definition of model input from the rest. It is, e.g., especially useful when you want to use the same model input for multiple models. 

If you want to do that, create a new configuration file `config_openai_input.yaml` with just the input template:
```yaml
input_template: # Template for input assembly.
  cls: MessagesTemplate  # name of class that is subclass of Template
  config: # configuration for defined class
    messages:   # List of message builders.
      - cls: OpenAIMultiModalMessageBuilder
        config:
          role: user
          content:
            - cls: OpenAITextContent  # name of class that is subclass of Content
              config: # configuration for defined class
                text: |
                  Describe the image in a few sentences. Describe the image itself (photo, drawing, graphs), possibly its style and historic period. Describe the image content, objects, actions, location, names.
                
                  Use the following metadata to help you:
                    document title: {{ doc_title }}
                    document author: {{ author }}
                    tags: {{ tags | join(", ") }}
            - cls: OpenAIImageContent  # name of class that is subclass of Content
              config: # configuration for defined class
                url: "{{image.filename}}"  # Text of the message.
                detail: LOW
```
The original configuration may remain the same, or you can simplify it by leaving the `input_template` field empty:
```yaml
convertor:  # Convertor to batch file.
  cls: ToOpenAIBatchFile  # name of class that is subclass of Convertor
  config: # configuration for defined class
    loader:  # Loader for the data.
      cls: HFImageLoader  # name of class that is subclass of Loader
      config: # configuration for defined class
        path_to: data # Path to the data.
        config: # Configuration name.
        split: train # Split of the dataset.
    id_format: "{{file_name}}" # Format string for custom id. You can use fields {{index}} and fields provided by the sample assembler.
    model: gpt-4o-mini # OpenAI model name.
    temperature: 1.0 # Temperature of the model.
    logprobs: false # Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.
    max_completion_tokens: 1024 # Maximum number of tokens generated.
    sample_assembler: # Sample assembler for API request.
      cls: ImageDatasetAssembler  # name of class that is subclass of APISampleAssembler
      config: # configuration for defined class
        input_template: # The input template will be loaded from the separate configuration file.
    response_format: # Format of the response.
```

You can then create the batch file using the following command:
```bash
aicaller create_batch_file --config config_openai.yaml --input_template_config config_openai_input.yaml > batch_openai.jsonl
```

This will replace the `input_template` in the `config_openai.yaml` with the one from `config_openai_input.yaml`.

### Ollama configuration
To create a batch file for Ollama we can follow a very similar process as for OpenAI. We will create a configuration file with the following command:
```bash
aicaller create_config --path config_ollama.yaml
```
Using the guide, we select the following options:
* create batch workflow
* ToOllamaBatchFile
* HFImageLoader
* ImageDatasetAssembler
* MessagesTemplate

Thus, almost the same steps as for OpenAI. The only difference is that we will use `ToOllamaBatchFile` convertor and `OllamaImageContent` content class.

We will edit the configuration file to look like this:
```yaml
convertor:  # Convertor to batch file.
  cls: ToOllamaBatchFile  # name of class that is subclass of Convertor
  config: # configuration for defined class
    loader:  # Loader for the data.
      cls: HFImageLoader  # name of class that is subclass of Loader
      config: # configuration for defined class
        path_to: data # Path to the data.
        config: # Configuration name.
        split: train # Split of the dataset.
    id_format: "{{file_name}}" # Format string for custom id. You can use fields {{index}} and fields provided by the sample assembler.
    model: llama3.2-vision # OpenAI model name.
    options: # additional model parameters listed in the documentation for the Modelfile such as temperature
      temperature: 1.0
      num_ctx: 2048
      num_predict: 128
    sample_assembler: # Sample assembler for API request.
      cls: ImageDatasetAssembler  # name of class that is subclass of APISampleAssembler
      config: # configuration for defined class
        input_template: # Template for input assembly.
          cls: MessagesTemplate  # name of class that is subclass of Template
          config: # configuration for defined class
            messages:  # List of message builders.
              - cls: OllamaMessageBuilder
                config:
                  role: user
                  content: |
                    Describe the image in a few sentences. Describe the image itself (photo, drawing, graphs), possibly its style and historic period. Describe the image content, objects, actions, location, names.
                  
                    Use the following metadata to help you:
                      document title: {{ doc_title }}
                      document author: {{ author }}
                      tags: {{ tags | join(", ") }}
                  images: [ "{{image.filename}}" ]
    format: # Format of the response.
```

## Sending the batch file
To send the batch file to the API we need firstly to create an API configuration file. We can do that by running configuration creation command:
```bash
aicaller create_config --path api.yaml
```

Select the following options:
* API
* OpenAPIFactory or OllamaAPIFactory (depending on which API you want to use)

This will create a configuration file that looks like this:
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
Fill it with your API key and change other fields as needed.

When you are done, you have several options for how to send the batch file to the API:

  * batch request (supported by OpenAI)
  * synchronous mode (`--synchronous`), which will send requests one by one and wait for the response for each of them.
  * asynchronous mode (`--asynchronous`), which will send requests in parallel and wait for the responses to come back. You can set `concurrency` in the API configuration file to limit the number of concurrent requests.

We will use the `--synchronous` mode for this example:
```bash
aicaller batch_request batch_openai.jsonl --config api.yaml --synchronous --results results_openai/ --only_output
```

You can see that we are using the `--only_output` flag, which will prevent printing additional information to the results. Also pay attention to the `--results` attribute, which specifies the results folder by using the slash at the end. This matters as in this case it will create a separate file for each request, in the results folder, named by the id of the request. If it was a path to a file without the slash at the end, it would create a single file with all the results in it.
All together, we will get single file for each request in the `results_openai` folder, with just the caption generated by the model.