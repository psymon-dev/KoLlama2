
[Read English](#kollama2--open-source-language-model-based-on-llama2-optimized-for-korean)

# KoLlama2 : 한국어에 최적화된 Llama2 기반 오픈소스 언어모델

KoLlama2(Korean Large Language Model Meta AI 2)는 영어 기반 LLM인 Llama2의 한국어 성능을 향상하기 위한 오픈소스 프로젝트입니다. 

<br/>

## 필요성

GPT3부터 Bert, Llama2에 이르기까지 대규모 언어모델의 놀라운 발전은 모든 이의 이목을 끌고 있습니다. 그러나 대규모 말뭉치를 사전학습하는 LLM의 특성상 학습 데이터 중 대다수는 영어로 구정되며, 한국어는 매우 적은 비율을 차지합니다.

<br/>

- GPT3의 사전학습 데이터 중 한국어 비율: 0.01697%
<p align="center" style="color:gray">
  <img style="margin:20px 0 10px 0" src="https://github.com/psymon-dev/KoLlama2/assets/91517542/79b72fee-3517-4a7e-a0a5-fda4c8f2a7ca" alt="image" width=482 />
  <br/>출처: 22p Table 10, Llama 2: Open Foundation and Fine-Tuned Chat Models, Hugo Touvron et al, July 18-2023.
</p> 

<br/>

- Llama2 모델의 사전학습 데이터 중 한국어 비율: 0.06%
<p align="center" style="color:gray">
  <img style="margin:20px 0 10px 0" src="https://github.com/psymon-dev/KoLlama2/assets/91517542/b50b9283-fb54-46b6-bc84-00bd363601c8" alt="image" width=482 />
  <br/>출처: https://github.com/openai/gpt-3/blob/master/dataset_statistics/languages_by_word_count.csv
</p> 

<br/>

이 비율은 전세계 인구(7.888 billion) 중 한국어 화자(81.7M) 비율(1.035%)과 비교해도 크게 낮은 수치입니다. 이는 고립어라는 한국어 특성, 준비되지 않은 한국어 말뭉치 등 여러 요인에 기반한 것이지만 결과적으로 한국어 사용자가 LLM의 풍부한 능력을 경험하는 것을 매우 제한하고 있습니다.

<br/>

## 기존 시도들
### 한국어기반 LLM 사전학습
가장 좋은 해결책 중 하나는 한국어 데이터로 사전학습한 자체 언어모델을 만드는 것입니다. 이러한 시도는 자본력을 갖춘 대기업의 주도로 진행되고 있습니다.

* Naver의 HyperCLOVA X : https://clova.ai/hyperclova
* Kakao의 KoGPT : https://github.com/kakaobrain/kogpt
* EleutherAI의 polyglot-ko : https://github.com/EleutherAI/polyglot

<br/>

이러한 접근법은 LLM의 한국어 능력 부족을 가장 확실하게 해결할 수 있습니다. 다만 문제는 LLM의 변화속도가 너무 빠르다는 데 있습니다. LLaMA 모델이 공개된 후 Llama2 모델이 공개되기까지 고작 5개월 밖에 걸리지 않았습니다. 매주 새로운 기술이 발표되는 현 상황에서 미래 발전 방향을 정확히 예측하거나, 매번 새로운 변화에 맞춰 대규모 언어모델을 학습하는 것은 불가능합니다.

따라서 우리는 자체 언어모델을 학습하는 것과 병행할 수 있는 더 가볍고 빠른 방법이 필요합니다.

<br/>

### 외국어기반 LLM 미세조정
외국어기반 LLM을 한국어로 미세조정하는 것은 이 문제에 대한 좋은 해결 책입니다. LLaMa 모델을 기반으로 아래와 같은 시도들이 있었습니다.

* KoAlpaca : https://github.com/Beomi/KoAlpaca
* KULLM : https://github.com/nlpai-lab/KULLM
* KoVicuna : https://github.com/melodysdreamj/KoVicuna
* KORani : https://github.com/krafton-ai/KORani

<br/>

이러한 시도들은 오픈소스 LLM에 대한 관심을 늘리고 다양한 미세조정 방법을 이해하는 데 도움을 주었지만 한계점도 명확했습니다.

<br/>

1. LLaMA 모델의 경우 사전학습 데이터에 한국어가 제외되어 Full-Finetuning, LoRA, QLoRA 등 어떤 방법으로도 만족스러운 한국어 성능을 내지 못했습니다.
<br/>

2. 통일된 학국어 학습 평가 방법이 없어 어떤 학습 방법이 가장 효과적인지 판단하기 어려웠습니다.
<br/>

3. 각 프로젝트가 개별 주체에 의해 산발로 전개되어 중복된 시도가 반복되었습니다.  

<br/>

## KoLlama2 프로젝트 제안
KoLlama2는 LLaMA 모델에서 얻은 경험을 바탕으로 외국어 기반 LLM을 한국어로 미세조정하는 가장 좋은 방법을 찾는 프로젝트입니다. 이를 위해 아래와 같은 시도들이 필요합니다.

1. QLoRA, LoRA, Full-Finetuning 등 다양한 방법론을 시도하여 Llama2에 포함된 0.01697% 한국어 능력이 얼마나 향상되는지 확인.
<br/>

2. Alpaca, Vicuna 등 다양한 데이터세트를 적용하여 어떤 형태 테이터세트가 한국어 능력향상에 가장 효과적인지 확인.
<br/>

3. 간단한 한영 번역부터 점차 난이도를 올리는 Curriculum Learning, 대규모 한국어 말뭉치로 사전학습 Step을 추가 학습, Chinese-LLaMA에서 사용한 어휘확장 등 새로운 기법들 시도.
<br/>

4. 각 방법론을 평가할 합리적 평가법 고안.


<br/>

## Benchmarks

## 참고 자료


<br/>

---

# KoLlama2 : Open source language model based on Llama2 optimized for Korean
KoLlama2 (Korean Large Language Model Meta AI 2) is an open-source project to improve the Korean performance of Llama2, an English-based LLM. 


## Problem

From GPT3 to Bert to Llama2, the amazing advances in large-scale language models have captured everyone's attention. However, due to the nature of LLMs pre-training on large corpora, the vast majority of training data is spoken in English, with Korean representing a very small percentage.

- Percentage of Korean in GPT3's pretraining data: 0.01697

<p align="center" style="color:gray">
  <img style="margin:20px 0 10px 0" src="https://github.com/psymon-dev/KoLlama2/assets/91517542/79b72fee-3517-4a7e-a0a5-fda4c8f2a7ca" alt="image" width=482 />
  <br/>22p Table 10, Llama 2: Open Foundation and Fine-Tuned Chat Models, Hugo Touvron et al, July 18-2023.
</p> 

- Percentage of Korean in the Llama2 model's pre-training data: 0.06%.

<p align="center" style="color:gray">
  <img style="margin:20px 0 10px 0" src="https://github.com/psymon-dev/KoLlama2/assets/91517542/b50b9283-fb54-46b6-bc84-00bd363601c8" alt="image" width=482 />
  <br/>https://github.com/openai/gpt-3/blob/master/dataset_statistics/languages_by_word_count.csv
</p> 

This percentage is significantly lower than the percentage of Korean speakers (81.7M) in the world's population (7.888 billion) (1.035%). This is based on a number of factors, including the isolated nature of Korean, an unprepared Korean corpus, and more, but the end result is that Korean speakers are severely limited in experiencing the richness of LLM.

## Problem Statement
### Korean-based LLM Pretrain

One of the best solutions is to create your own language model, pre-trained with Korean data. This is being done by large, well-funded companies.

* Naver HyperCLOVA X : https://clova.ai/hyperclova
* Kakao KoGPT : https://github.com/kakaobrain/kogpt
* EleutherAI polyglot-ko : https://github.com/EleutherAI/polyglot

This approach would most certainly address the LLM's lack of Korean language skills. The problem is that the LLM is changing so fast. It took only five months between the release of the LLaMA model and the release of the Llama2 model. With new technologies being released every week, it's impossible to accurately predict future developments, or to train a large language model to adapt to each new change.

Therefore, we need a lighter and faster method that can be used in parallel with learning our own language models.

### Fine-tuning a English-based LLM
Fine-tuning a foreign language-based LLM into Korean is a good solution to this problem. The following attempts have been made based on the LLaMa model.


* KoAlpaca : https://github.com/Beomi/KoAlpaca
* KULLM : https://github.com/nlpai-lab/KULLM
* KoVicuna : https://github.com/melodysdreamj/KoVicuna
* KORani : https://github.com/krafton-ai/KORani

While these attempts have increased interest in open source LLMs and helped me understand the various ways to fine-tune them, the limitations are clear.

1. For the LLaMA model, Korean was excluded from the pre-training data, so no method, including Full-Finetuning, LoRA, and QLoRA, could produce satisfactory Korean performance.

2. There was no unified method for evaluating Korean language learning, making it difficult to determine which learning method was most effective.

3. Each project was developed sporadically by individual entities, resulting in redundant attempts.  

## KoLlama2 Project Suggested
KoLlama2 is a project to find the best way to fine-tune a English-based LLM into Korean based on the experience gained from the LLaMA model. To achieve this, the following attempts are required.

1. try different methodologies such as QLoRA, LoRA, and Full-Finetuning to see how much the 0.01697% Korean proficiency included in Llama2 improves.

2. Apply various datasets such as Alpaca and Vicuna to see which type of dataset is most effective for improving Korean proficiency.

3. try new techniques such as curriculum learning that gradually increases the difficulty from simple English to Korean translation, additional pre-learning steps with a large Korean corpus, and vocabulary expansion used in Chinese-LLaMA.

4. devising a reasonable evaluation method to assess each methodology.

## Benchmarks

## References

# Llama 2

We are unlocking the power of large language models. Our latest version of Llama is now accessible to individuals, creators, researchers and businesses of all sizes so that they can experiment, innovate and scale their ideas responsibly. 

This release includes model weights and starting code for pretrained and fine-tuned Llama language models — ranging from 7B to 70B parameters.

This repository is intended as a minimal example to load [Llama 2](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/) models and run inference. For more detailed examples leveraging HuggingFace, see [llama-recipes](https://github.com/facebookresearch/llama-recipes/).

## Download

⚠️ **7/18: We're aware of people encountering a number of download issues today. Anyone still encountering issues should remove all local files, re-clone the repository, and [request a new download link](https://ai.meta.com/resources/models-and-libraries/llama-downloads/). It's critical to do all of these in case you have local corrupt files. When you receive the email, copy *only* the link text - it should begin with https://download.llamameta.net and not with https://l.facebook.com, which will give errors.**



In order to download the model weights and tokenizer, please visit the [Meta AI website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and accept our License.

Once your request is approved, you will receive a signed URL over email. Then run the download.sh script, passing the URL provided when prompted to start the download. Make sure that you copy the URL text itself, **do not use the 'Copy link address' option** when you right click the URL. If the copied URL text starts with: https://download.llamameta.net, you copied it correctly. If the copied URL text starts with: https://l.facebook.com, you copied it the wrong way.

Pre-requisites: make sure you have `wget` and `md5sum` installed. Then to run the script: `./download.sh`.

Keep in mind that the links expire after 24 hours and a certain amount of downloads. If you start seeing errors such as `403: Forbidden`, you can always re-request a link.

### Access on Hugging Face

We are also providing downloads on [Hugging Face](https://huggingface.co/meta-llama). You must first request a download from the Meta AI website using the same email address as your Hugging Face account. After doing so, you can request access to any of the models on Hugging Face and within 1-2 days your account will be granted access to all versions.

## Setup

In a conda env with PyTorch / CUDA available, clone the repo and run in the top-level directory:

```
pip install -e .
```

## Inference

Different models require different model-parallel (MP) values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 70B    | 8  |

All models support sequence length up to 4096 tokens, but we pre-allocate the cache according to `max_seq_len` and `max_batch_size` values. So set those according to your hardware.

### Pretrained Models

These models are not finetuned for chat or Q&A. They should be prompted so that the expected answer is the natural continuation of the prompt.

See `example_text_completion.py` for some examples. To illustrate, see command below to run it with the llama-2-7b model (`nproc_per_node` needs to be set to the `MP` value):

```
torchrun --nproc_per_node 1 example_text_completion.py \
    --ckpt_dir llama-2-7b/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 128 --max_batch_size 4
```

### Fine-tuned Chat Models

The fine-tuned models were trained for dialogue applications. To get the expected features and performance for them, a specific formatting defined in [`chat_completion`](https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212)
needs to be followed, including the `INST` and `<<SYS>>` tags, `BOS` and `EOS` tokens, and the whitespaces and breaklines in between (we recommend calling `strip()` on inputs to avoid double-spaces).

You can also deploy additional classifiers for filtering out inputs and outputs that are deemed unsafe. See the llama-recipes repo for [an example](https://github.com/facebookresearch/llama-recipes/blob/main/inference/inference.py) of how to add a safety checker to the inputs and outputs of your inference code.

Examples using llama-2-7b-chat:

```
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 4
```

Llama 2 is a new technology that carries potential risks with use. Testing conducted to date has not — and could not — cover all scenarios.
In order to help developers address these risks, we have created the [Responsible Use Guide](Responsible-Use-Guide.pdf). More details can be found in our research paper as well.

## Issues

Please report any software “bug,” or other problems with the models through one of the following means:
- Reporting issues with the model: [github.com/facebookresearch/llama](http://github.com/facebookresearch/llama)
- Reporting risky content generated by the model: [developers.facebook.com/llama_output_feedback](http://developers.facebook.com/llama_output_feedback)
- Reporting bugs and security concerns: [facebook.com/whitehat/info](http://facebook.com/whitehat/info)

## Model Card
See [MODEL_CARD.md](MODEL_CARD.md).

## License

Our model and weights are licensed for both researchers and commercial entities, upholding the principles of openness. Our mission is to empower individuals, and industry through this opportunity, while fostering an environment of discovery and ethical AI advancements. 

See the [LICENSE](LICENSE) file, as well as our accompanying [Acceptable Use Policy](USE_POLICY.md)

## References

1. [Research Paper](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
2. [Llama 2 technical overview](https://ai.meta.com/resources/models-and-libraries/llama)
3. [Open Innovation AI Research Community](https://ai.meta.com/llama/open-innovation-ai-research-community/)

## Original LLaMA
The repo for the original llama release is in the [`llama_v1`](https://github.com/facebookresearch/llama/tree/llama_v1) branch.
