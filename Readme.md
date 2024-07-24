# 并没有那么魔的魔改版单机Llama2

这是一个基于[Llama 2 官方仓库](git@github.com:meta-llama/llama.git)魔改的Llama2推理代码。去除了为了多机或多卡分布式训练的部分，适合单机、单卡、单节点运行。

# Prerequisites

以下内容是实测可行的配置：
1. 如果你使用Nvidia GPU + CUDA
    - 你需要手动安装支持CUDA的Pytorch, 理论上Pytorch 2.0+都可以，但经测试2.2.2是没有问题的。可以参见[官方文档](https://pytorch.org/get-started/previous-versions/#wheel-1)安装带CUDA支持的Pytorch 2.2.2
    - 通常运行起来，在`max_seq_len`（即输入的prompt的token数+生成的token数的最大值）== 512，`max_batch_size`（即一次最多可以同时推理几套对话，通常类似ChatGPT的一个聊天算一套对话，无论消息多少条，最后会算这套对话的总token数）的配置下，需要大约15G显存。

2. 如果你使用CPU推理：
    - 你需要把main里面的`torch.set_default_device()`函数改一下
    - 在我的Linux上使用了大约26.5G的内存


# Quick Start

首先安装依赖
- 除了python3.9+以外，唯二的依赖是`sentencepiece`（谷歌开源的分词tokenize）器和`pytorch`
- `sentencepice`非常好装只要`pip install sentencepice`就可以了。
- pytorch需要根据官网Start Locally页面安装符合你机器需要的最新版本 或 https://pytorch.org/get-started/previous-versions 中的文档安装指定版本
- 经测试Ubuntu 20.04以上可以直接`pip install -r requirements.txt`

其次是需要根据Meta的要求同意Llama的License并下载Llama2模型，通常你需要克隆官方仓库，然后运行官方仓库中的下载脚本。考虑到消费级硬件通常的极限，你可以考虑只下载7b的模型。假设官方仓库的路径在`<llama2_dir>`, 根据你电脑的配置设置一下`max_seq_len`和`max_batch_size`。

最后是运行`src/main`

```bash
python src/main.py <llama2_dir>
```

或者你也可以直接进main函数自己该llama2_dir

# 修改Prompt
直接进main.py里那一大段字符串里改就好，或者在我的基础上继续魔改成可以通过Web UI或命令行的方式聊天。

# 进一步魔改

初步魔改可以改变跑不跑得起来的问题，因为它直接与资源相关：
1. 改模型
默认用的7B-chat版本，你可以改成别的版本。在
2. 改batch size和seq len
默认值是`max_seq_len=512`和`max_batch_size=6`, 这两个决定了用来存token的Tensor和模型内部的kv cache大小，这两个大小很大程度上影响显存或内存占用

# 疯狂魔改

TBD
