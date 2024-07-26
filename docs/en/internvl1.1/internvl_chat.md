# InternVL-Chat-V1-1

## Introduction

We released [🤗 InternVL-Chat-V1-1](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-1), featuring a structure similar to LLaVA, including a ViT, an MLP projector, and an LLM.
As shown in the figure below, we connected our InternViT-6B to LLaMA2-13B through a simple MLP projector. Note that the LLaMA2-13B used here is not the original model but an internal chat version obtained by incrementally pre-training and fine-tuning the LLaMA2-13B base model for Chinese language tasks. Overall, our model has a total of 19 billion parameters.

<p align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/64119264f0f81eb569e0d569/HD29tU-g0An9FpQn1yK8X.png" style="width: 75%;">
</p>

In this version, we explored increasing the resolution to 448 × 448, enhancing OCR capabilities, and improving support for Chinese conversations. Since the 448 × 448 input image generates 1024 visual tokens after passing through the ViT, leading to a significant computational burden, we use a pixel shuffle (unshuffle) operation to reduce the 1024 tokens to 256 tokens.

For more detailed information about this model, please read our [blog](https://internvl.github.io/blog/2024-01-24-InternVL-1.1/).

## Model Preparation

| model name         | type | download                                                          |  size   |
| ------------------ | ---- | ----------------------------------------------------------------- | :-----: |
| InternVL-Chat-V1-1 | MLLM | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-1) | 35.0 GB |

Please download the above model weights and place them in the `pretrained/` folder.

```sh
cd pretrained/
# pip install -U huggingface_hub
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL-Chat-V1-1 --local-dir InternVL-Chat-V1-1
```

The directory structure is:

```sh
pretrained
└── InternVL-Chat-V1-1
```

## Performance

|             model              |  LLaVA-1.5   | InternVL-Chat-V1-0 | InternVL-Chat-V1-0 | InternVL-Chat-V1-1 |
| :----------------------------: | :----------: | :----------------: | :----------------: | :----------------: |
|           resolution           |     336      |        336         |        448         |        448         |
|         vision encoder         | CLIP-L-336px | InternViT-6B-224px | InternViT-6B-448px | InternViT-6B-448px |
|         language model         |  Vicuna-13B  |     Vicuna-13B     |     Vicuna-13B     |     LLaMA2-13B     |
|                                |              |                    |                    |                    |
|    VQAv2<sub>testdev</sub>     |     80.0     |        80.2        |        82.0        |        80.9        |
|     GQA<sub>testdev</sub>      |     63.3     |        63.9        |        64.1        |        62.5        |
|     VizWiz<sub>test</sub>      |     53.6     |        54.6        |        60.1        |        57.3        |
|       SQA<sub>test</sub>       |     71.6     |        70.1        |        71.6        |        90.1        |
| TextVQA<sub>val, w/o OCR</sub> |      -       |         -          |         -          |        64.2        |
| TextVQA<sub>val, w/ OCR</sub>  |     61.3     |        58.7        |        64.8        |        68.6        |
|              POPE              |     85.9     |        87.1        |        87.2        |        87.1        |
|    MME<sub>perception</sub>    |    1531.3    |       1546.9       |       1579.0       |       1659.8       |
|     MMB-EN<sub>test</sub>      |     67.7     |        66.5        |        68.2        |        75.4        |
|     MMB-CN<sub>test</sub>      |     63.6     |        61.9        |        64.0        |        70.3        |
|   MMVet<sub>GPT-4-0613</sub>   |     35.4     |        33.7        |        36.7        |        46.7        |

- Note that we use the [official evaluation server](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator) to test the MMVet scores, with `GPT-4-0613` serving as the judge model. Using different versions of GPT-4 as the judge can result in significant score variations.

Here, we have conducted only a simple performance comparison. For more detailed performance information and additional evaluation metrics, please refer to our [performance summary table](<>).

## Quick Start

We provide an example code to run InternVL-Chat-V1-1 using `transformers`.

We also welcome you to experience the InternVL2 series models in our [online demo](https://internvl.opengvlab.com/).

> Please use transformers==4.37.2 to ensure the model works normally.

### Model Loading

#### 16-bit (bf16 / fp16)

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL-Chat-V1-1"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
```

#### BNB 8-bit Quantization

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL-Chat-V1-1"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval()
```

#### BNB 4-bit Quantization

> **⚠️ Warning:** Due to significant quantization errors with BNB 4-bit quantization on InternViT-6B, the model may produce nonsensical outputs and fail to understand images. Therefore, please avoid using BNB 4-bit quantization.

#### Multiple GPUs

The reason for writing the code this way is to avoid errors that occur during multi-GPU inference due to tensors not being on the same device. By ensuring that the first and last layers of the large language model (LLM) are on the same device, we prevent such errors.

```python
import math
import torch
from transformers import AutoTokenizer, AutoModel

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {'InternVL-Chat-V1-1': 40}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

path = "OpenGVLab/InternVL-Chat-V1-1"
device_map = split_model('InternVL-Chat-V1-1')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map=device_map).eval()
```

### Inference with Transformers

#### Pure-text conversation

```python
from transformers import AutoTokenizer, AutoModel
import torch

path = "OpenGVLab/InternVL-Chat-V1-1"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

generation_config = dict(max_new_tokens=1024, do_sample=False)
question = 'Hello, who are you?'
response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')

question = 'Can you tell me a story?'
response, history = model.chat(tokenizer, None, question, generation_config, history=history, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')
```

#### Single-image single-round conversation

```python
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image
import torch

path = "OpenGVLab/InternVL-Chat-V1-1"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

image_processor = CLIPImageProcessor.from_pretrained(path)
image = Image.open('./examples/image2.jpg').resize((448, 448))
pixel_values = image_processor(images=image, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()

generation_config = dict(max_new_tokens=1024, do_sample=False)
question = '<image>\nPlease describe the image shortly.'
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f'User: {question}')
print(f'Assistant: {response}')
```

#### Single-image multi-round conversation

```python
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image
import torch

path = "OpenGVLab/InternVL-Chat-V1-1"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

image_processor = CLIPImageProcessor.from_pretrained(path)
image = Image.open('./examples/image2.jpg').resize((448, 448))
pixel_values = image_processor(images=image, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()

generation_config = dict(max_new_tokens=1024, do_sample=False)
question = '<image>\nPlease describe the image in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')

question = 'Please write a poem according to the image.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')
```

#### Multi-image multi-round conversation, combined images

> **⚠️️ Warning:** Please note that for this model, we support multi-image chat in the interface, but the results are not very good due to the lack of training with multi-image data.

```python
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image
import torch

path = "OpenGVLab/InternVL-Chat-V1-1"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

image_processor = CLIPImageProcessor.from_pretrained(path)
image1 = Image.open('./examples/image1.jpg').resize((448, 448))
pixel_values1 = image_processor(images=image1, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()
image2 = Image.open('./examples/image2.jpg').resize((448, 448))
pixel_values2 = image_processor(images=image2, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

generation_config = dict(max_new_tokens=1024, do_sample=False)
question = '<image>\nDescribe the two images in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               history=None, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')

question = 'What are the similarities and differences between these two images.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               history=history, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')
```

#### Multi-image multi-round conversation, separate images

> **⚠️️ Warning:** Please note that for this model, we support multi-image chat in the interface, but the results are not very good due to the lack of training with multi-image data.

```python
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image
import torch

path = "OpenGVLab/InternVL-Chat-V1-1"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

image_processor = CLIPImageProcessor.from_pretrained(path)
image1 = Image.open('./examples/image1.jpg').resize((448, 448))
pixel_values1 = image_processor(images=image1, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()
image2 = Image.open('./examples/image2.jpg').resize((448, 448))
pixel_values2 = image_processor(images=image2, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

generation_config = dict(max_new_tokens=1024, do_sample=False)
question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')

question = 'What are the similarities and differences between these two images.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=history, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')
```

#### Batch inference, single image per sample

```python
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image
import torch

path = "OpenGVLab/InternVL-Chat-V1-1"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

image_processor = CLIPImageProcessor.from_pretrained(path)
image1 = Image.open('./examples/image1.jpg').resize((448, 448))
pixel_values1 = image_processor(images=image1, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()
image2 = Image.open('./examples/image2.jpg').resize((448, 448))
pixel_values2 = image_processor(images=image2, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

generation_config = dict(max_new_tokens=1024, do_sample=False)
questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
responses = model.batch_chat(tokenizer, pixel_values,
                             num_patches_list=num_patches_list,
                             questions=questions,
                             generation_config=generation_config)
for question, response in zip(questions, responses):
    print(f'User: {question}')
    print(f'Assistant: {response}')
```

#### Video multi-round conversation

> **⚠️️ Warning:** Please note that for this model, we support video chat in the interface, but the results are not very good due to the lack of training with video data.

```python
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import torch


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    image_processor = CLIPImageProcessor.from_pretrained(path)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB').resize((448, 448))
        pixel_values = image_processor(images=img, return_tensors='pt').pixel_values
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


path = "OpenGVLab/InternVL-Chat-V1-1"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

generation_config = dict(max_new_tokens=1024, do_sample=False)

video_path = './examples/red-panda.mp4'
pixel_values, num_patches_list = load_video(video_path, num_segments=8)
pixel_values = pixel_values.to(torch.bfloat16).cuda()
video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
question = video_prefix + 'What is the red panda doing?'
# Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')

question = 'Describe this video in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=history, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')
```

#### Streaming output

Besides this method, you can also use the following code to get streamed output.

```python
from transformers import TextIteratorStreamer
from threading import Thread

# Initialize the streamer
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10)
# Define the generation configuration
generation_config = dict(max_new_tokens=1024, do_sample=False, streamer=streamer)
# Start the model chat in a separate thread
thread = Thread(target=model.chat, kwargs=dict(
    tokenizer=tokenizer, pixel_values=pixel_values, question=question,
    history=None, return_history=False, generation_config=generation_config,
))
thread.start()

# Initialize an empty string to store the generated text
generated_text = ''
# Loop through the streamer to get the new text as it is generated
for new_text in streamer:
    if new_text == model.conv_template.sep:
        break
    generated_text += new_text
    print(new_text, end='', flush=True)  # Print each new chunk of generated text on the same line
```

## Evaluation

To evaluate the performance of the InternVL-Chat-V1-1 model across various tasks, follow the instructions for each specific dataset. Ensure that the appropriate number of GPUs is allocated as specified.

> 1⃣️ We simultaneously use InternVL and VLMEvalKit repositories for model evaluation. For certain datasets like MMVet and LLaVA-Bench, different GPT-4 versions used as judges cause significant result discrepancies between two codebases.

> 2⃣️ Please note that evaluating the same model using different testing toolkits like InternVL and VLMEvalKit can result in slight differences, which is normal. Updates to code versions and variations in environment and hardware can also cause minor discrepancies in results.

> 3⃣️️ Note, the dataset description is generated by GPT-4 and may contain errors.

### Evaluation using InternVL Codebase

#### Data Preparation

Please prepare the evaluation data according to the [guidance provided here](../get_started/eval_data_preparation.md).

#### MME

MME is a comprehensive benchmark designed to evaluate Multimodal Large Language Models (MLLMs) on both perception and cognition abilities across 14 different subtasks, ensuring robust and diverse testing of these models.

Please use the following command to perform the test with 1 GPU:

```bash
GPUS=1 sh evaluate.sh pretrained/InternVL-Chat-V1-1 mme
```

The expected test results are:

```
=========== Perception ===========
total score: 1664.5088035214085

         existence  score: 185.0
         count  score: 173.33333333333334
         position  score: 163.33333333333334
         color  score: 190.0
         posters  score: 161.22448979591837
         celebrity  score: 149.11764705882354
         scene  score: 153.5
         landmark  score: 167.5
         artwork  score: 144.0
         OCR  score: 177.5


=========== Cognition ===========
total score: 360.7142857142857

         commonsense_reasoning  score: 130.71428571428572
         numerical_calculation  score: 70.0
         text_translation  score: 110.0
         code_reasoning  score: 50.0
```

#### OKVQA

OKVQA (Outside Knowledge Visual Question Answering) is a dataset designed for visual question answering tasks that require external knowledge beyond what is visible in the image, featuring over 14,000 questions to evaluate the reasoning abilities of AI models.

Please use the following command to perform the test with 8 GPU:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 vqa-okvqa-val
```

The expected test results are:

```
okvqa_val 0.6406262386048285
```

#### TextVQA

TextVQA is a dataset designed to evaluate visual question answering models by requiring them to read and reason about text present within images, containing 45,336 questions over 28,408 images from the OpenImages dataset.

The TextVQA dataset provides official OCR results, specifically Rosetta OCR tokens. During testing with InstructBLIP and LLaVA 1.5, the OCR results are input to the LLM as a prompt. If you want to input Rosetta OCR tokens, use the following command:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 vqa-textvqa-val-ocr
```

The expected test results are:

```
textvqa_val_ocr 0.686240000000003
```

If you do not want to input Rosetta OCR tokens, use this command:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 vqa-textvqa-val
```

The expected test results are:

```
textvqa_val 0.6420000000000028
```

#### VizWiz

The VizWiz VQA dataset is a visual question answering dataset created to help answer visual questions posed by blind individuals. It contains over 31,000 visual questions, where users took a picture using a mobile phone and recorded a spoken question about it. Each question comes with 10 crowdsourced answers. This dataset addresses tasks such as predicting the answer to a visual question and determining whether a visual question can be answered.

For the validation set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 vqa-vizwiz-val
```

The expected test results are:

```
vizwiz_val 0.5899899435054417
```

For the test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 vqa-vizwiz-test
```

For the test set, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/2185/overview).

The expected test results are:

```
57.3
```

#### ChartQA

The ChartQA dataset is a comprehensive benchmark for question answering about charts that involves both visual and logical reasoning. It includes a mix of 9.6K human-written questions and 23.1K machine-generated questions derived from chart summaries. This dataset is designed to evaluate models that can understand and analyze charts by answering complex questions that often require multiple logical and arithmetic operations, as well as referencing visual features of the charts.

The ChartQA dataset includes two test sets: `chartqa_test_human` and `chartqa_test_augmented`. The final score for model evaluation is calculated as the average of the scores on these two test sets:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 vqa-chartqa-test
```

The expected test results are:

```
['chartqa_test_human', {'relaxed_accuracy': 0.4034}]
['chartqa_test_augmented', {'relaxed_accuracy': 0.795}]
average score = (40.34 + 79.5) / 2 = 59.9
```

#### DocVQA

The DocVQA dataset consists of 50,000 questions on 12,000+ document images. It is designed for visual question answering tasks where questions are answered using text within the document images. The dataset includes OCR transcriptions and ground truth answers, supporting evaluation of models that interpret and extract information from documents.

For the validation set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 vqa-docvqa-val
```

The expected test results are:

```
Overall ANLS: 0.476
```

For the test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 vqa-docvqa-test
```

For the test set, submit the results to the [evaluation server](https://rrc.cvc.uab.es/?ch=17).

The expected test results are:

```
Overall ANLS: 0.481
```

#### AI2D

The AI2D dataset contains over 5,000 grade school science diagrams with extensive annotations and 15,000 multiple-choice questions for research on diagram understanding and question answering.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 vqa-ai2d-test
```

The expected test results are:

```
ai2diagram_test {'accuracy': 0.7240140932642487}
```

#### InfographicVQA

The InfographicVQA dataset is a collection of infographics accompanied by natural language questions and answers. This dataset includes a diverse range of infographics sourced from thousands of different websites, ensuring a variety of layouts and designs. It comprises 30,035 questions across 5,485 images, split into training, validation, and test sets.

For the validation set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 vqa-infovqa-val
```

The expected test results are:

```
Overall ANLS: 0.3334
```

For the test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 vqa-infovqa-test
```

For the test set, submit the results to the [evaluation server](https://rrc.cvc.uab.es/?ch=17).

The expected test results are:

```
Overall ANLS: 0.320
```

#### GQA

The GQA dataset is a large-scale visual question answering dataset designed for real-world visual reasoning and compositional question answering. It contains over 22 million questions grounded in real images, each accompanied by detailed scene graphs that describe objects, their attributes, and relationships within the scene. The dataset includes images from the Visual Genome dataset, with questions that require various reasoning skills such as spatial understanding and multi-step inference.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 vqa-gqa-testdev
```

The expected test results are:

```
Accuracy: 62.46%
```

#### ScienceQA

The ScienceQA dataset is a large-scale benchmark for multimodal science question answering, consisting of 21,208 multiple-choice questions derived from elementary and high school science curricula. This dataset features a diverse range of topics across natural science, social science, and language science. It includes questions with image context (48.7%), text context (48.2%), and both (30.8%).

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 scienceqa
```

The expected test results are:

```
Acc@1: 0.90133019335647
```

#### POPE

The POPE (Polling-based Object Probing Evaluation) dataset is designed to evaluate object hallucination in MLLMs. The dataset consists of 3,000 questions related to the captions of 500 images. By treating the MLLMs' answers to these questions as a binary classification task, the dataset allows researchers to measure accuracy, precision, recall, and F1 scores to determine the extent of hallucination in the models.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 pope
```

The expected test results are:

```
Category: random, # samples: 2910
TP      FP      TN      FN
1196    14      1396    304
Accuracy: 0.8907216494845361
Precision: 0.9884297520661157
Recall: 0.7973333333333333
F1 score: 0.8826568265682657
Yes ratio: 0.41580756013745707
0.883, 0.891, 0.988, 0.797, 0.416
====================================
Category: popular, # samples: 3000
TP      FP      TN      FN
1196    47      1453    304
Accuracy: 0.883
Precision: 0.9621882542236525
Recall: 0.7973333333333333
F1 score: 0.8720379146919431
Yes ratio: 0.41433333333333333
0.872, 0.883, 0.962, 0.797, 0.414
====================================
Category: adversarial, # samples: 3000
TP      FP      TN      FN
1196    89      1411    304
Accuracy: 0.869
Precision: 0.930739299610895
Recall: 0.7973333333333333
F1 score: 0.858886894075404
Yes ratio: 0.42833333333333334
0.859, 0.869, 0.931, 0.797, 0.428
====================================

(0.883 + 0.872 + 0.859) / 3 = 87.1
```

#### Tiny LVLM

The Tiny LVLM-eHub is a streamlined evaluation benchmark designed to assess the multimodal capabilities of MLLMs, including models like Bard. It focuses on six categories of multimodal abilities: visual perception, visual knowledge acquisition, visual reasoning, visual commonsense, object hallucination, and embodied intelligence.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 tiny_lvlm
```

The expected test results are:

```
Visual_Knowledge_Acquisition: 0.74
Object_Hallucination: 0.8966666666666666
Visual_Commonsense: 0.6
Visual_Perception: 0.574
Visual_Reasoning: 0.6218181818181818
Overall: 3.4324848484848486
```

#### MMMU

The MMMU dataset is a comprehensive benchmark designed to evaluate multimodal models on college-level tasks that require domain-specific knowledge and reasoning. It includes 11,500 questions sourced from college exams, quizzes, and textbooks, spanning six disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions cover 30 subjects and feature 30 types of images, such as charts, diagrams, maps, tables, and more.

For the validation set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 mmmu-val
```

The expected test results are:

```
{'Overall-Art and Design': {'num': 120, 'acc': 0.558}, 'Art': {'num': 30, 'acc': 0.633}, 'Art_Theory': {'num': 30, 'acc': 0.7}, 'Design': {'num': 30, 'acc': 0.633}, 'Music': {'num': 30, 'acc': 0.267}, 'Overall-Business': {'num': 150, 'acc': 0.313}, 'Accounting': {'num': 30, 'acc': 0.333}, 'Economics': {'num': 30, 'acc': 0.4}, 'Finance': {'num': 30, 'acc': 0.133}, 'Manage': {'num': 30, 'acc': 0.433}, 'Marketing': {'num': 30, 'acc': 0.267}, 'Overall-Science': {'num': 150, 'acc': 0.333}, 'Biology': {'num': 30, 'acc': 0.367}, 'Chemistry': {'num': 30, 'acc': 0.3}, 'Geography': {'num': 30, 'acc': 0.267}, 'Math': {'num': 30, 'acc': 0.4}, 'Physics': {'num': 30, 'acc': 0.333}, 'Overall-Health and Medicine': {'num': 150, 'acc': 0.393}, 'Basic_Medical_Science': {'num': 30, 'acc': 0.367}, 'Clinical_Medicine': {'num': 30, 'acc': 0.433}, 'Diagnostics_and_Laboratory_Medicine': {'num': 30, 'acc': 0.4}, 'Pharmacy': {'num': 30, 'acc': 0.367}, 'Public_Health': {'num': 30, 'acc': 0.4}, 'Overall-Humanities and Social Science': {'num': 120, 'acc': 0.542}, 'History': {'num': 30, 'acc': 0.567}, 'Literature': {'num': 30, 'acc': 0.767}, 'Sociology': {'num': 30, 'acc': 0.4}, 'Psychology': {'num': 30, 'acc': 0.433}, 'Overall-Tech and Engineering': {'num': 210, 'acc': 0.29}, 'Agriculture': {'num': 30, 'acc': 0.433}, 'Architecture_and_Engineering': {'num': 30, 'acc': 0.267}, 'Computer_Science': {'num': 30, 'acc': 0.233}, 'Electronics': {'num': 30, 'acc': 0.333}, 'Energy_and_Power': {'num': 30, 'acc': 0.2}, 'Materials': {'num': 30, 'acc': 0.233}, 'Mechanical_Engineering': {'num': 30, 'acc': 0.333}, 'Overall': {'num': 900, 'acc': 0.388}}
```

For the test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 mmmu-test
```

Then submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/2179/overview). The expected test results are:

```
All subject resultes
{'Overall-Art & Design': {'num': 1163, 'acc': 0.537}, 'Art': {'num': 231, 'acc': 0.606}, 'Art_Theory': {'num': 429, 'acc': 0.59}, 'Design': {'num': 169, 'acc': 0.746}, 'Music': {'num': 334, 'acc': 0.314}, 'Overall-Business': {'num': 1428, 'acc': 0.317}, 'Accounting': {'num': 380, 'acc': 0.345}, 'Economics': {'num': 267, 'acc': 0.345}, 'Finance': {'num': 355, 'acc': 0.245}, 'Manage': {'num': 245, 'acc': 0.29}, 'Marketing': {'num': 181, 'acc': 0.392}, 'Overall-Science': {'num': 2426, 'acc': 0.282}, 'Biology': {'num': 345, 'acc': 0.357}, 'Chemistry': {'num': 603, 'acc': 0.244}, 'Geography': {'num': 565, 'acc': 0.312}, 'Math': {'num': 505, 'acc': 0.275}, 'Physics': {'num': 408, 'acc': 0.24}, 'Overall-Health & Medicine': {'num': 1752, 'acc': 0.365}, 'Basic_Medical_Science': {'num': 326, 'acc': 0.436}, 'Clinical_Medicine': {'num': 325, 'acc': 0.397}, 'Diagnostics_and_Laboratory_Medicine': {'num': 162, 'acc': 0.364}, 'Pharmacy': {'num': 430, 'acc': 0.309}, 'Public_Health': {'num': 509, 'acc': 0.346}, 'Overall-Humanities & Social Science': {'num': 947, 'acc': 0.564}, 'History': {'num': 278, 'acc': 0.579}, 'Literature': {'num': 112, 'acc': 0.804}, 'Sociology': {'num': 252, 'acc': 0.556}, 'Psychology': {'num': 305, 'acc': 0.469}, 'Overall-Tech & Engineering': {'num': 2784, 'acc': 0.28}, 'Agriculture': {'num': 287, 'acc': 0.369}, 'Architecture_and_Engineering': {'num': 551, 'acc': 0.272}, 'Computer_Science': {'num': 371, 'acc': 0.315}, 'Electronics': {'num': 256, 'acc': 0.152}, 'Energy_and_Power': {'num': 432, 'acc': 0.306}, 'Materials': {'num': 458, 'acc': 0.26}, 'Mechanical_Engineering': {'num': 429, 'acc': 0.27}, 'Overall': {'num': 10500, 'acc': 0.353}}

Leaderboard
[{'test_split': {'Art & Design': 0.537, 'Business': 0.317, 'Science': 0.282, 'Health & Medicine': 0.365, 'Humanities & Social Science': 0.564, 'Tech & Engineering': 0.28, 'Overall': 0.353}}]
```

#### MMVet (GPT-4-0613)

> **⚠️ Warning:** Here, we use `GPT-4-0613` as the judge model, while in VLMEvalKit, `GPT-4-Turbo` is used as the judge model. Using different versions of GPT-4 can result in significant score variations. Therefore, testing the same model with the two codebases can lead to notable score differences.

The MM-Vet dataset is a comprehensive benchmark designed to evaluate the integrated capabilities of MLLMs. It encompasses six core vision-language (VL) capabilities: recognition, knowledge, optical character recognition (OCR), spatial awareness, language generation, and math. The dataset includes 200 images and 218 questions, each requiring one or more of these capabilities to answer. The evaluation uses an open-ended LLM-based approach, allowing assessment across various answer styles and question types.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 mmvet
```

Then, submit the results to the [evaluation server](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator). The expected test results are:

```
runs: [46.7]
```

#### MMBench

The MMBench dataset is a comprehensive multi-modality benchmark designed to evaluate the fine-grained abilities of vision-language models. It contains around 3,000 multiple-choice questions covering 20 ability dimensions, structured into a hierarchical taxonomy. These dimensions include perception and reasoning abilities, further broken down into specific skills like coarse and fine-grained perception, attribute reasoning, and logic reasoning.

For the English dev / test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 mmbench-dev-en
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 mmbench-test-en

```

Then, submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission). The expected test results are:

```
mmbench-dev-en: 76.7
mmbench-test-en: 75.4
```

For the Chinese dev / test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 mmbench-dev-cn
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 mmbench-test-cn

```

Then, submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission). The expected test results are:

```
mmbench-dev-cn: 71.9
mmbench-test-cn: 70.3
```

#### CCBench

CCBench, a multi-modal benchmark in the domain of Chinese Culture, is designed to evaluate the performance of MLLMs on tasks specifically related to Chinese cultural content.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 ccbench-dev
```

Then, submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission). The expected test results are:

```
ccbench-dev: 43.3
```

#### SEED

CCBench is a multimodal benchmark specifically designed to evaluate models on tasks related to Chinese culture. It is part of the larger MMBench suite of benchmarks, developed by the OpenCompass Community, and aims to provide fine-grained evaluations across various capabilities of vision-language models. CCBench includes 510 questions in a multiple-choice format, focusing on cultural knowledge and understanding.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 seed
```

The expected test results are:

```
Data type Scene Understanding: 78.63%
Data type Instance Identity: 77.44%
Data type Instance Location: 74.66%
Data type Instance Attributes: 70.04%
Data type Instances Counting: 65.79%
Data type Spatial Relation: 58.90%
Data type Instance Interaction: 77.32%
Data type Visual Reasoning: 79.15%
Data type Text Understanding: 39.53%
Data type Action Recognition: 54.34%
Data type Action Prediction: 40.82%
Data type Procedure Understanding: 37.24%
Total accuracy: 67.40%
+ Image accuracy: 73.24%
Video accuracy: 45.28%
```

#### MMVP

The MMVP dataset is designed to benchmark the performance of multimodal large language models (MLLMs) in visual question answering tasks. This dataset focuses on identifying "CLIP-blind pairs," which are images that appear similar to the CLIP model despite having clear visual differences. The MMVP dataset includes 300 images derived from ImageNet-1k and LAION-Aesthetics, each paired with straightforward questions to evaluate the models' visual capabilities. It highlights the challenges these systems face, often leading to incorrect responses and hallucinated explanations.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 mmvp
```

The expected test results are:

```
Evaluating MMVP ...
Results saved to results/MMVP_240725163208.jsonl
The accuracy is 0.4466666666666667
```

#### LLaVA-Bench (GPT-4-0613)

> **⚠️ Warning:** Here, we use `GPT-4-0613` as the judge model, while in VLMEvalKit, `GPT-4-Turbo` is used as the judge model. Using different versions of GPT-4 can result in significant score variations. Therefore, testing the same model with the two codebases can lead to notable score differences.

The LLaVA-Bench-in-the-Wild dataset is designed to evaluate the capabilities of MLLMs in handling more complex and diverse visual tasks. It includes a set of 24 images with 60 associated questions, covering a range of indoor and outdoor scenes, memes, paintings, and sketches. Each image is paired with detailed, manually curated descriptions and questions that test the model's generalizability to novel domains.

```bash
export OPENAI_API_KEY='your openai key'
GPUS=1 sh evaluate.sh pretrained/InternVL-Chat-V1-1 llava-bench
```

The expected test results are:

```
all *72.8* 87.7 63.8
llava_bench_complex [8.75, 6.643] 75.9
llava_bench_complex 75.9 87.5 66.4
llava_bench_conv [9.0, 6.118] 68.0
llava_bench_conv 68.0 90.0 61.2
llava_bench_detail [8.533, 6.2] 72.7
llava_bench_detail 72.7 85.3 62.0
```

#### RefCOCO / RefCOCO+ / RefCOCO-g

RefCOCO, RefCOCO+, and RefCOCOg are datasets used for tasks involving referring expression comprehension, segmentation, and generation. These datasets are built upon the MSCOCO dataset, and they are essential for evaluating models in natural language processing and computer vision.

```bash
GPUS=8 sh evalulate.sh pretrained/InternVL-Chat-V1-1 refcoco
```

The expected test results are:

```
RefCOCO val, 84.7
RefCOCO testA, 89.9
RefCOCO testB, 78.6
RefCOCO+ val, 78.5
RefCOCO+ testA, 85.6
RefCOCO+ testB, 70.1
RefCOCO‑g val, 81.0
RefCOCO‑g test, 81.4
```

#### MVBench

MVBench is a comprehensive multimodal video understanding benchmark developed to evaluate the temporal comprehension capabilities of MLLMs. It includes 20 challenging video tasks that require temporal understanding and cannot be effectively solved using a single frame. The benchmark uses a novel static-to-dynamic method, transforming static tasks into dynamic ones to systematically generate video tasks that demand a wide range of temporal skills, from perception to cognition.

We evaluate our models on MVBench by extracting 16 frames from each video, and each frame was resized to a 448x448 image.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-1 mvbench
```

The expected test results are:

```
{"Action Sequence": 62.5, "Action Prediction": 56.49999999999999, "Action Antonym": 49.0, 
"Fine-grained Action": 41.0, "Unexpected Action": 64.5, "Object Existence": 49.0, 
"Object Interaction": 64.0, "Object Shuffle": 32.5, "Moving Direction": 39.5, 
"Action Localization": 27.500000000000004, "Scene Transition": 88.5, "Action Count": 42.0,
"Moving Count": 33.0, "Moving Attribute": 57.99999999999999, "State Change": 46.5, 
"Fine-grained Pose": 44.0, "Character Order": 59.0, "Egocentric Navigation": 38.5,
"Episodic Reasoning": 44.5, "Counterfactual Inference": 39.0, "Avg": 48.949999999999996}
```

### Evaluation using VLMEvalKit Codebase

#### Data Preparation

VLMEvalKit will automatically download the data for evaluation, so you do not need to prepare it manually.

#### MathVista

The MathVista dataset is a comprehensive benchmark for evaluating mathematical reasoning within visual contexts. It consists of three newly created datasets—IQTest, FunctionQA, and PaperQA—designed to address logical reasoning on puzzle test figures, algebraic reasoning over functional plots, and scientific reasoning with academic paper figures, respectively.

```bash
torchrun --nproc-per-node=8 run.py --data MathVista_MINI --model InternVL-Chat-V1-1 --verbose
```

The expected test results are:

```
--  ---------------------------  ----  ---  ---  -------  -------
 0  Overall                      1000  594  363  59.4     36.3
 1  scientific reasoning          122   95   58  77.8689  47.541
 2  textbook question answering   158  107   74  67.7215  46.8354
 3  numeric commonsense           144   67   56  46.5278  38.8889
 4  arithmetic reasoning          353  143  122  40.5099  34.5609
 5  visual question answering     179  102   80  56.9832  44.6927
 6  geometry reasoning            239  198   63  82.8452  26.3598
 7  algebraic reasoning           281  217   74  77.2242  26.3345
 8  geometry problem solving      208  181   50  87.0192  24.0385
 9  math word problem             186   74   66  39.7849  35.4839
10  logical reasoning              37   25    5  67.5676  13.5135
11  figure question answering     269  130   93  48.3271  34.5725
12  statistical reasoning         301  126  109  41.8605  36.2126
--  ---------------------------  ----  ---  ---  -------  -------
```

#### HallusionBench

HallusionBench is a comprehensive benchmark designed to evaluate image-context reasoning in MLLMs, focusing on identifying issues related to language hallucination and visual illusion. The dataset consists of 346 images paired with 1,129 questions crafted by human experts. These questions are divided into two categories: Visual Dependent (VD) and Visual Supplement (VS), allowing the benchmark to assess the nuanced understanding and interpretation of visual data by MLLMs.

```bash
torchrun --nproc-per-node=8 run.py --data HallusionBench --model InternVL-Chat-V1-1 --verbose
```

The expected test results are:

```
"split","aAcc","fAcc","qAcc"
"Overall","56.256572029442694","26.011560693641616","26.153846153846157"
"VD","54.483925549915405","29.565217391304348","23.465703971119133"
"VS","59.166666666666664","18.96551724137931","30.337078651685395"
"VD_figure","70.0","51.21951219512195","41.02564102564102"
"VD_ocr","75.28089887640449","53.48837209302325","51.162790697674424"
"VD_math","47.22222222222222","5.555555555555555","18.51851851851852"
"VS_ocr","53.70370370370371","23.076923076923077","11.11111111111111"
"VS_map","54.6875","13.636363636363635","12.5"
"VS_chart","62.30769230769231","17.5","44.73684210526316"
"VD_video","42.35294117647059","10.416666666666668","5.797101449275362"
"VD_illusion","52.77777777777778","27.419354838709676","18.055555555555554"
"VS_table","60.71428571428571","21.428571428571427","30.23255813953488"

result = (56.256572029442694 + 26.011560693641616 + 26.153846153846157) / 3 = 36.1
```

#### MMStar

The MMStar dataset is an advanced multimodal benchmark designed to evaluate the capabilities of MLLMs. It comprises 1,500 carefully selected samples that are balanced and purified to ensure they exhibit visual dependency and minimal data leakage. The dataset evaluates models across six core capabilities and 18 detailed axes, focusing on complex multimodal tasks that require advanced reasoning and understanding of visual content.

```bash
torchrun --nproc-per-node=8 run.py --data MMStar --model InternVL-Chat-V1-1 --verbose
```

The expected test results are:

```
"split","Overall","coarse perception","fine-grained perception","instance reasoning","logical reasoning","math","science & technology"
"none","0.452","0.652","0.384","0.612","0.404","0.292","0.368"
```

#### OCRBench

OCRBench is a comprehensive evaluation benchmark designed to assess the OCR capabilities of MLLMs. It includes five components: Text Recognition, Scene Text-Centric Visual Question Answering (VQA), Document-Oriented VQA, Key Information Extraction (KIE), and Handwritten Mathematical Expression Recognition (HMER). The benchmark encompasses data from 29 datasets, making it one of the most thorough OCR evaluation tools available. OCRBench aims to reveal both the strengths and weaknesses of MLLMs, particularly in handling multilingual text, handwritten text, non-semantic text, and mathematical expressions. The benchmark includes 1,000 question-answer pairs, all manually verified for precision.

```bash
torchrun --nproc-per-node=8 run.py --data OCRBench --model InternVL-Chat-V1-1 --verbose
```

The expected test results are:

```
{
    "Text Recognition": 230,
    "Scene Text-centric VQA": 157,
    "Doc-oriented VQA": 72,
    "Key Information Extraction": 71,
    "Handwritten Mathematical Expression Recognition": 0,
    "Final Score": 530,
    "Final Score Norm": 53.0
}
```

#### MMMU

The MMMU dataset is a comprehensive benchmark designed to evaluate multimodal models on college-level tasks that require domain-specific knowledge and reasoning. It includes 11,500 questions sourced from college exams, quizzes, and textbooks, spanning six disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions cover 30 subjects and feature 30 types of images, such as charts, diagrams, maps, tables, and more.

```bash
torchrun --nproc-per-node=8 run.py --data MMMU_DEV_VAL --model InternVL-Chat-V1-1 --verbose
```

The expected test results are:

```
-----------------------------------  -------------------  -------------------
split                                validation           dev
Overall                              0.4022222222222222   0.47333333333333333
Accounting                           0.36666666666666664  0.2
Agriculture                          0.4666666666666667   0.2
Architecture_and_Engineering         0.23333333333333334  0.4
Art                                  0.6333333333333333   0.8
Art_Theory                           0.7333333333333333   0.6
Basic_Medical_Science                0.4                  0.6
Biology                              0.36666666666666664  0.4
Chemistry                            0.4666666666666667   0.8
Clinical_Medicine                    0.4                  0.4
Computer_Science                     0.26666666666666666  0.6
Design                               0.6333333333333333   0.6
Diagnostics_and_Laboratory_Medicine  0.4                  0.8
Economics                            0.43333333333333335  0.8
Electronics                          0.4                  0.4
Energy_and_Power                     0.2                  0.2
Finance                              0.3333333333333333   0.8
Geography                            0.26666666666666666  0.0
History                              0.5333333333333333   0.6
Literature                           0.7666666666666667   0.4
Manage                               0.4666666666666667   0.2
Marketing                            0.26666666666666666  0.8
Materials                            0.16666666666666666  0.2
Math                                 0.43333333333333335  0.4
Mechanical_Engineering               0.26666666666666666  0.2
Music                                0.26666666666666666  0.2
Pharmacy                             0.4                  0.6
Physics                              0.3333333333333333   0.2
Psychology                           0.43333333333333335  0.6
Public_Health                        0.36666666666666664  0.4
Sociology                            0.36666666666666664  0.8
Art & Design                         0.5666666666666667   0.55
Business                             0.37333333333333335  0.56
Health & Medicine                    0.3933333333333333   0.56
Humanities & Social Science          0.525                0.6
Science                              0.37333333333333335  0.36
Tech & Engineering                   0.2857142857142857   0.3142857142857143
-----------------------------------  -------------------  -------------------
```

#### RealWorldQA

The RealWorldQA dataset is a benchmark designed to evaluate the real-world spatial understanding capabilities of multimodal AI models. It consists of over 700 images, each accompanied by a question and a verifiable answer, focusing on various real-world scenarios, including those captured from vehicles. This dataset aims to test how well AI models comprehend physical environments and spatial relations, enhancing their ability to interpret and analyze real-world scenes.

```bash
torchrun --nproc-per-node=8 run.py --data RealWorldQA --model InternVL-Chat-V1-1 --verbose
```

The expected test results are:

```
"split","Overall"
"none","0.5803921568627451"
```

#### MMVet (GPT-4-Turbo)

The MM-Vet dataset is a comprehensive benchmark designed to evaluate the integrated capabilities of MLLMs. It encompasses six core vision-language (VL) capabilities: recognition, knowledge, optical character recognition (OCR), spatial awareness, language generation, and math. The dataset includes 200 images and 218 questions, each requiring one or more of these capabilities to answer. The evaluation uses an open-ended LLM-based approach, allowing assessment across various answer styles and question types.

```bash
torchrun --nproc-per-node=8 run.py --data MMVet --model InternVL-Chat-V1-1 --verbose
```

The expected test results are:

```
2024-07-26 17:28:10,536 - RUN - INFO - The evaluation of model InternVL-Chat-V1-1 x dataset MMVet has finished!
2024-07-26 17:28:10,536 - RUN - INFO - Evaluation Results:
2024-07-26 17:28:10,538 - RUN - INFO -
-  -------  ---  -------
0  rec      187  49.9465
1  ocr      108  42.5
2  know      84  36.4286
3  gen       80  36.25
4  spat      75  42.9333
5  math      26  22.3077
6  Overall  218  44.7706
-  -------  ---  -------
```

#### LLaVA-Bench (GPT-4-Turbo)

The LLaVA-Bench-in-the-Wild dataset is designed to evaluate the capabilities of MLLMs in handling more complex and diverse visual tasks. It includes a set of 24 images with 60 associated questions, covering a range of indoor and outdoor scenes, memes, paintings, and sketches. Each image is paired with detailed, manually curated descriptions and questions that test the model's generalizability to novel domains.

```bash
torchrun --nproc-per-node=8 run.py --data LLaVABench --model InternVL-Chat-V1-1 --verbose
```

The expected test results are:

```
2024-07-25 16:11:27,640 - RUN - INFO - The evaluation of model InternVL-Chat-V1-1 x dataset LLaVABench has finished!
2024-07-25 16:11:27,640 - RUN - INFO - Evaluation Results:
2024-07-25 16:11:27,641 - RUN - INFO -
-  -------  ----  ----  ----
0  overall *64.8* 46.7  72
1  complex  64.4  47.1  73.2
2  conv     67.1  57.6  85.9
3  detail   61.7  33.3  54
-  -------  ----  ----  ----
```

## Citation

If you find this project useful in your research, please consider citing:

```BibTeX
@article{chen2023internvl,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and Li, Bin and Luo, Ping and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2312.14238},
  year={2023}
}
@article{chen2024far,
  title={How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites},
  author={Chen, Zhe and Wang, Weiyun and Tian, Hao and Ye, Shenglong and Gao, Zhangwei and Cui, Erfei and Tong, Wenwen and Hu, Kongzhi and Luo, Jiapeng and Ma, Zheng and others},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
}
```

<br>
<br>
