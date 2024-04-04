## 使用 InternLM2-Chat-1.8B 模型生成 300 字的小故事
创建实验环境
```bash
conda create -n demo python==3.10.13
```
![创建环境](https://openi.pcl.ac.cn/komisteng/homework/raw/branch/master/InternLM/%e5%b1%8f%e5%b9%95%e6%88%aa%e5%9b%be%202024-04-03%20233914.png)
启动环境
```bash
conda activate demo
```
装库，装包
```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install huggingface-hub==0.17.3
pip install transformers==4.34 
pip install psutil==5.9.8
pip install accelerate==0.24.1
pip install streamlit==1.32.2 
pip install matplotlib==3.8.3 
pip install modelscope==1.9.5
pip install sentencepiece==0.1.99
```

下载模型
```bash
mkdir -p /root/demo
touch /root/demo/cli_demo.py
touch /root/demo/download_mini.py
cd /root/demo
python /root/demo/download_mini.py
```

启动模型，让模型输出一个300字的小故事
```python
python cli_demo.py
User  >>>  请帮我创作一个300字的小故事，主题不限
```
