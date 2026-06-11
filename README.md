<h1 align="center">JaiTTS: A Thai Voice Cloning Model</h1>

<p align="center">
  <a href="https://jaitts-demo.jts.co.th/">
    <img src="https://img.shields.io/badge/Try%20the%20Live%20Demo-jaitts--demo.jts.co.th-00A86B?style=for-the-badge" alt="Try the live demo" height="36">
  </a>
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2604.27607">
    <img src="https://img.shields.io/badge/Paper-arXiv-red?style=for-the-badge" alt="Paper">
  </a>
  <a href="https://huggingface.co/JTS-AI">
    <img src="https://img.shields.io/badge/Hugging%20Face-JTS--AI-yellow?style=for-the-badge" alt="Hugging Face">
  </a>
  <a href="https://jts.co.th/jai">
    <img src="https://img.shields.io/badge/Website-JAI-orange?style=for-the-badge" alt="Website">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="License">
  </a>
</p>

<p align="center">
  <img src="asset/JaiTTS-logo.png" alt="JaiTTS overview" width="250">
</p>

This repository contains the benchmark and the code used for benchmarking.

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Human Evaluation](#human-evaluation)
- [Demo](#demo)
- [Quick Start](#quick-start)
- [Links](#links)
- [Citation](#citation)

## Overview

JaiTTS-v1.0 focuses on strong Thai intelligibility, robust long-duration generation, speaker preservation, and reliable handling of real-world Thai inputs.

Main contributions from the paper:

- Strong Thai zero-shot voice cloning for both short and long utterances
- Direct synthesis from raw text, including numerals and Thai-English code-switching, without explicit text normalization
- A dedicated Thai benchmark with short (`1-15s`) and long (`16-30s`) evaluation tracks
- Fast inference with a Real-Time Factor (RTF) of `0.1136`

## Key Results

> `JaiTTS-v1.0` reaches `1.94%` CER on short-form Thai voice cloning, slightly better than the `1.98%` human reference in the paper's benchmark.

> On long-form generation, `JaiTTS-v1.0` reaches `2.55%` CER, close to the `2.47%` human reference and clearly ahead of the evaluated baselines reported in the paper.

> In human preference testing, `JaiTTS-v1.0` wins `283` of `400` pairwise comparisons against commercial flagship systems, with `59` ties and `58` losses.

### Objective Benchmark

| Model | Short CER (%) ↓ | Short SIM ↑ | Long CER (%) ↓ | Long SIM ↑ |
| :-- | --: | --: | --: | --: |
| Human (Ground Truth) | 1.98 | 0.61 | 2.47 | 0.83 |
| Qwen3-TTS-0.6B | 3.14 | 0.62 | 6.10 | 0.79 |
| Qwen3-TTS-1.7B | 2.56 | 0.62 | 3.64 | 0.78 |
| ThonburianTTS | 6.26 | 0.48 | -- | -- |
| Moss-TTS-v1.5 | 4.05 | 0.57 | 4.39 | 0.76 |
| Omnivoice | 2.73 | 0.65 | 6.28 | **0.82** |
| VoxCPM2 | 4.98 | **0.68** | 3.37 | 0.80 |
| Kaitom Voice V3 (Alpha) | 2.34 | 0.59 | 5.81 | 0.79 |
| **[JaiTTS-v1.0](https://jaitts-demo.jts.co.th/)** | **1.94** | 0.62 | **2.55** | 0.76 |

### Inference Speed

| Model | RTF ↓ |
| :-- | --: |
| Qwen3-TTS-0.6B | 1.5092 |
| Qwen3-TTS-1.7B | 1.5409 |
| ThonburianTTS | 0.1150 |
| **JaiTTS-v1.0** | **0.1136** |

## Human Evaluation

The paper reports a blind side-by-side evaluation with:

- `20` native Thai evaluators
- `30` speakers (`15` female, `15` male)
- Reference prompts of about `10-13s`
- `400` total pairwise comparisons against commercial systems

| Comparison | JaiTTS-v1.0 Wins | Ties | Competitor Wins |
| :-- | --: | --: | --: |
| vs. eleven_v3 | 161 | 19 | 20 |
| vs. speech-2.8-hd | 122 | 40 | 38 |
| **Total** | **283** | **59** | **58** |

## Demo

https://github.com/user-attachments/assets/8e44c106-d3b5-4ecd-bccd-8325cbe54489

## Quick Start

This evaluation code is adapted from the [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval) 

Install dependencies:

```bash
pip install -r requirements.txt
```

If your ASR model access requires authentication, export a Hugging Face token first:

```bash
export HF_TOKEN=<your_token_here>
```

Before running evaluation:

- Download the short Thai benchmark test set: [Link](https://drive.google.com/file/d/1syOkdb0G2Etsi1xG-xT1p6_EDPpoXMqD/view?usp=sharing)
- Download the finetuned WavLM checkpoint for similarity evaluation: [Link](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view)

Evaluation models used by the scripts:

- *CER*: [`typhoon-ai/typhoon-whisper-large-v3`](https://huggingface.co/typhoon-ai/typhoon-whisper-large-v3) for Thai transcription
- *SIM*: `wavlm_large` fine-tuned for speaker verification, loaded from the WavLM checkpoint above

`meta.lst` format:

```text
filename|prompt_text|prompt_wav|infer_text|normalized_infer_text
```

For Thai evaluation, the expected 5-column format is:

- `filename`: the synthesized wav filename without extension
- `prompt_text`: the text spoken in the prompt audio
- `prompt_wav`: the prompt audio path
- `infer_text`: the raw text to synthesize
- `normalized_infer_text`: the normalized ground-truth text used for evaluation

Thai handling is selected explicitly via the `lang` argument passed to the evaluation scripts.

Expected synthesized audio layout:

- Each sample must exist at `{output_dir}/{filename}.wav`
- Relative `prompt_wav` paths are resolved relative to the directory containing `meta.lst`

Run Thai evaluations:

```bash
# WER
bash cal_wer.sh {the path of the meta.lst file} {the directory of synthesized audio} th
# SIM
bash cal_sim.sh {the path of the meta.lst file} {the directory of synthesized audio} {the path of wavlm_large_finetune.pth} th
```

## Links

- Demo: <https://jaitts-demo.jts.co.th/>
- Paper: <https://arxiv.org/pdf/2604.27607>
- Website: <https://jts.co.th/jai>
- Hugging Face: <https://huggingface.co/JTS-AI>
- Contact: <jts.ai.team@gmail.com>

## Citation
```
@misc{karnjanaekarin2026jaittsthaivoicecloning,
      title={JaiTTS: A Thai Voice Cloning Model}, 
      author={Jullajak Karnjanaekarin and Pontakorn Trakuekul and Narongkorn Panitsrisit and Sumana Sumanakul and Vichayuth Nitayasomboon and Nithid Guntasin and Thanavin Denkavin and Attapol T. Rutherford},
      year={2026},
      eprint={2604.27607},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.27607}, 
}
```
