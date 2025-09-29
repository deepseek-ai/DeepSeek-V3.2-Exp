# DeepSeek-V3.2-Exp

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="DeepSeek-V3" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://chat.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20V3-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/deepseek-ai" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://discord.gg/Tc7c45Zzu5" target="_blank" style="margin: 2px;">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/qr.jpeg?raw=true" target="_blank" style="margin: 2px;">
    <img alt="Wechat" src="https://img.shields.io/badge/WeChat-DeepSeek%20AI-brightgreen?logo=wechat&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank" style="margin: 2px;">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

## Introduction


We are excited to announce the official release of DeepSeek-V3.2-Exp, an experimental version of our model. As an intermediate step toward our next-generation architecture, V3.2-Exp builds upon V3.1-Terminus by introducing DeepSeek Sparse Attention—a sparse attention mechanism designed to explore and validate optimizations for training and inference efficiency in long-context scenarios.

This experimental release represents our ongoing research into more efficient transformer architectures, particularly focusing on improving computational efficiency when processing extended text sequences.

<div align="center">
 <img src="cost.jpg" >
</div>

- DeepSeek Sparse Attention (DSA) achieves fine-grained sparse attention for the first time, delivering substantial improvements in long-context training and inference efficiency while maintaining virtually identical model output quality.


- To rigorously evaluate the impact of introducing sparse attention, we deliberately aligned the training configurations of DeepSeek-V3.2-Exp with V3.1-Terminus. Across public benchmarks in various domains, DeepSeek-V3.2-Exp demonstrates performance on par with V3.1-Terminus.


| Benchmark | DeepSeek-V3.2 | DeepSeek-V3.1-Terminus |
| :--- | :---: | :---: |
| **Reasoning Mode w/o Tool Use** | | |
| MMLU-Pro | 85.0 | 85.0 |
| GPQA-Diamond | 79.9 | 80.7 |
| Humanity's Last Exam | 19.8 | 21.7 |
| LiveCodeBench | 74.1 | 74.9 |
| AIME 2025 | 89.3 | 88.4 |
| HMMT 2025 | 83.6 | 86.1 |
| Codeforces | 2121 | 2046 |
| Aider-Polyglot | 74.5 | 76.1 |
| **Agentic Tool Use** | | |
| BrowseComp | 40.1 | 38.5 |
| BrowseComp-zh | 47.9 | 45.0 |
| SimpleQA | 97.1 | 96.8 |
| SWE Verified | 67.8 | 68.4 |
| SWE-bench Multilingual | 57.9 | 57.8 |
| Terminal-bench | 37.7 | 36.7 |





## How to Run Locally

We provide an updated inference demo code in the [inference](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference) folder to help the community quickly get started with our model and understand its architectural details.

First convert huggingface model weights to the the format required by our inference demo. Set `MP` to match your available GPU count:
```bash
cd inference
export EXPERTS=256
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP}
```

Launch the interactive chat interface and start exploring DeepSeek's capabilities:
```bash
export CONFIG=config_671B_v3.2.json
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --interactive
```

## Run with vLLM

vLLM supports day-0 support of DeepSeek-V3.2-Exp. See the [recipes](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html) for up-to-date details.

## License

This repository and the model weights are licensed under the [MIT License](LICENSE).

## Citation

```
@misc{deepseekai2024deepseekv32,
      title={DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention}, 
      author={DeepSeek-AI},
      year={2025},
}
```

## Contact

If you have any questions, please raise an issue or contact us at [service@deepseek.com](service@deepseek.com).
