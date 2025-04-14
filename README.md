# AICaller
Package for using API models. It is designed mainly for experimenting with various models. It allows to easily process Hugging Face datasets, or JSONL/CSV datasets, and send requests to OpenAI compatibles APIs with just using simple configuration files.

## Installation

```bash
pip install aicaller
```


## Usage
The package is designed a batch oriented way. It means that you must first create a batch file with API requests, and then you can send the requests to the API. 

Using this two stage approach allows you to check and save the raw requests that are sent to the API.

**If you prefer learning by doing, you can check the examples folder.**

## Batch file creation
Firstly, you need to create a batch file, fo that you can use the `create_batch_file` argument that expects a configuration `--config` (see config [creation](#configuration)) and voluntary a `--path` to file with data to be processed.

```bash
aicaller create_batch_file --config config.yaml --path data.jsonl > batch.jsonl
```

## Batch Split
It might be necessary to split the batch into smaller batches. It expects following arguments:

* `file` - path to batch file
* `output` - path to folder where the split files will be saved
* `max_tokens` - maximum number of tokens in one batch

```bash
aicaller split_batch batch.jsonl splits 1000000
```

## Sending requests
After your batch file is created, you can send the requests to the API (see API config [creation](#configuration)). You can use the `batch_request` command. Here is an example of how to use it:

```bash
aicaller batch_request batch.jsonl -c api_config.yaml -r results.jsonl
```

To see all available options, you can use the `--help` argument:

```bash
aicaller batch_request --help
```

## Configuration
There are two types of configuration files that you can use: one for creating batch files and one for sending requests to the API.

If you want to create a new configuration file, please use the `create_config` command, which will lead you through the process of creating a new configuration file:

```bash
aicaller create_config --path config.yaml
```

### Batch file configuration
The batch file configuration is a YAML file that defines how to create the batch file from a dataset. Here we will describe multiple options that you can use in an order that they appear during the configuration creation process. Detailed description of each attribute is always available directly in the configuration file.

#### Convertor
There are following convertors available:

* ToOpenAIBatchFile
  * Allows to create a batch file for OpenAI compatible APIs
* ToOllamaBatchFile
  * Even though Ollama API is compatible for basic usage with OpenAI API, it is not compatible with all features. Thus, we suggest to use this convertor for Ollama API.

### Loaders
This package allows to load Hugging Face datasets, or JSONL/CSV datasets, using following loaders:

* JSONLLoader
  * Loads JSONL files using Hugging Face dataset loader
* CSVLoader
  * Loads CSV files using Hugging Face dataset loader
* HFLoader
  * Loads text oriented Hugging Face datasets
* HFImageLoader
  * Loads image oriented Hugging Face datasets

### Sample Assemblers
A sample assembler is a component responsible for creating a sample from loaded data. There are two assemblers one for text and one for images:
* TextDatasetAssembler
* ImageDatasetAssembler

### Templates
Template specifies the format and content of a sample. It can be simple as string with Jinja2 template, or it can be whole chat history.

Now this package supports following types of templates:

* StringTemplate
  * Simple string template, that allows to use Jinja2 template
* MessagesTemplate
  * Allows to define whole chat history with roles and text/image content.
  * As the specification of messages varies for different APIs. There are different types of message builders.


### API configuration
File for configuring API connection. It is used for defining API key, URL, and other parameters.

