# CS336: Assignment 1 + MiniTrain (Assignment 2)

本仓库包含两部分：根目录是 Assignment 1 的模型、tokenizer 与训练实现；
[assignment2-systems](./assignment2-systems/README.md) 是在此基础上扩展的 MiniTrain 系统项目，
包含单卡训练、注意力后端对比、DDP、ZeRO-1、教学版 FSDP、checkpoint 与文本生成。

MiniTrain 已在 RTX 3050 Laptop GPU 上训练 TinyStories 至 12000 步，
并提供[实验结果与复现命令](./assignment2-systems/README.md)。
数据集与大型 checkpoint 不存放在 Git 仓库；小型实验日志保存在
`assignment2-systems/results/`。

## Assignment 1 原始说明

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Assignment 1 的实现与测试适配器已在本仓库完成；可直接运行上面的测试命令。

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

