#### 1、配置环境
没什么好说的，注意配置好cuda环境即可


#### 2、数据集准备

```shell
mkdir -p /root/ft && cd /root/ft
mkdir -p /root/ft/data && cd /root/ft/data
touch /root/ft/data/generate_data.py
python /root/ft/data/generate_data.py
touch /root/ft/tree.py
python /root/ft/tree.py /root/ft/data
ln -s /root/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b /root/ft/model
```

少量数据微调还是非常简单操作的，关键是如何生产单轮或者多轮对话，其实有ultrachat论文的生成方法。完全可以利用起来。

#### 3、配置文件

虽然我们用的数据集并不是 alpaca 而是我们自己通过脚本制作的小助手数据集 ，但是由于我们是通过 QLoRA 的方式对 internlm2-chat-1.8b 进行微调。而最相近的配置文件应该就是 internlm2_1_8b_qlora_alpaca_e3 ，因此我们可以选择拷贝这个配置文件到当前目录：

xtuner copy-cfg internlm2_1_8b_qlora_alpaca_e3 /root/ft/config

假如我们真的打开配置文件后，我们可以看到整体的配置文件分为五部分：

- PART 1 Settings：涵盖了模型基本设置，如预训练模型的选择、数据集信息和训练过程中的一些基本参数（如批大小、学习率等）。
- PART 2 Model & Tokenizer：指定了用于训练的模型和分词器的具体类型及其配置，包括预训练模型的路径和是否启用特定功能（如可变长度注意力），这是模型训练的核心组成部分。

- PART 3 Dataset & Dataloader：描述了数据处理的细节，包括如何加载数据集、预处理步骤、批处理大小等，确保了模型能够接收到正确格式和质量的数据。

- PART 4 Scheduler & Optimizer：配置了优化过程中的关键参数，如学习率调度策略和优化器的选择，这些是影响模型训练效果和速度的重要因素。

- PART 5 Runtime：定义了训练过程中的额外设置，如日志记录、模型保存策略和自定义钩子等，以支持训练流程的监控、调试和结果的保存。

一般来说我们需要更改的部分其实只包括前三部分，而且修改的主要原因是我们修改了配置文件中规定的模型、数据集。后两部分都是 XTuner 官方帮我们优化好的东西，一般而言只有在魔改的情况下才需要进行修改。下面我们将根据项目的要求一步步的进行修改和调整吧！

#### 4、启动训练

```shell
# 指定保存路径
xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train
```

```shell
# 使用 deepspeed 来加速训练
xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train_deepspeed --deepspeed deepspeed_zero2
```

另外假如我们模型中途中断了，我们也可以参考以下方法实现模型续训工作
假如我们的模型训练过程中突然被中断了，我们也可以通过在原有指令的基础上加上 --resume {checkpoint_path} 来实现模型的继续训练。需要注意的是，这个继续训练得到的权重文件和中断前的完全一致，并不会有任何区别。下面我将用训练了500轮的例子来进行演示。
```shell
# 模型续训
xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train --resume /root/ft/train/iter_600.pth
```

#### 5、整合lora模型
```shell
# 创建一个保存转换后 Huggingface 格式的文件夹
mkdir -p /root/ft/huggingface

# 模型转换
# xtuner convert pth_to_hf ${配置文件地址} ${权重文件地址} ${转换后模型保存地址}
xtuner convert pth_to_hf /root/ft/train/internlm2_1_8b_qlora_alpaca_e3_copy.py /root/ft/train/iter_768.pth /root/ft/huggingface
```

```shell
xtuner convert pth_to_hf /root/ft/train/internlm2_1_8b_qlora_alpaca_e3_copy.py /root/ft/train/iter_768.pth /root/ft/huggingface --fp32 --max-shard-size 2GB
```

```shell
# 创建一个名为 final_model 的文件夹存储整合后的模型文件
mkdir -p /root/ft/final_model

# 解决一下线程冲突的 Bug 
export MKL_SERVICE_FORCE_INTEL=1

# 进行模型整合
# xtuner convert merge  ${NAME_OR_PATH_TO_LLM} ${NAME_OR_PATH_TO_ADAPTER} ${SAVE_PATH} 
xtuner convert merge /root/ft/model /root/ft/huggingface /root/ft/final_model
```

```shell
touch ~/ft/web_demo/InternLM/chat/web_demo2.py
```


