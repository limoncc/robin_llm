#### 1.RAG是什么？
RAG（Retrieval Augmented Generation）的中文为检索增强生成技术，结合了检索（R）与生成（G）的技术，这个结合的过程增强了语言模型在处理特定任务时的性能，从外部知识元检索相关信息，并将这些信息用于指导语言模型生成更准确、丰富的回答。

可以理解为，RAG为搜索引擎，将用户输入内容作为索引，在外部知识库中搜索相关内容，结合大预言模型能力生成回答。RAG最大特点：

2.RAG的特点
解决大模型处理知识密集任务时遇到的各种挑战，如生成幻觉（hallucination）、过时知识、缺乏透明和可追溯的推理过程。

RAG能让大模型实现外部记忆，能够解决许多大模型常见问题的同时，提供更准确的回答。由于没有训练过程，总体成本就会很低。

3.RAG的应用
问答系统、文本生成系统、信息检索，以及结合多模态大模型后能够对图片进行描述。

4.RAG工作原理
经典RAG由三个部分构成——索引（indexing）、检索（retrieval）、生成（generation）。

索引部分负责处理外部知识，检索部分负责接受问题，生成部分将检索到的内容与原始问题一起做为提示输入到大模型中。

#### 二、RAG常见优化方法

1、嵌入优化 Embedding Optimization: 结合稀疏和密集检索多任务   
2、索引优化 Indexing Optimization: 细粒度分割(Chunk)元数据  
3、查询优化 Query Optimization: 查询扩张、转化、多查询
4、上下问管理 Context Curation: 重排（rerank）上下文选择/压缩
5、迭代检索 Iterative Retrieval: 根据初始查询和迄今为止生成的文本进行重复搜索
6、递归检索 Recursive Retrieval:  迭代细化搜索查询，链式推理指导检索过程
7、自适应检索 Adaptive Retrieval: Flare, Self-RAG 使用LLMs主动决定检索的最佳时机和内容
8、LLM微调 LLM Fine-tuning: 检索微调、生成微调、双重微调

#### 三、在 InternLM Studio 上部署茴香豆技术助手

茴香豆技术助手一些值得学习的地方

1、前置了一个意图识别的模块主要包括：
- 主题提取
- 问题相关性
- 问题与检索内容相关性
- 安全拒答机制

这是它的提示词，大家可以看看
```python
if self.language == 'zh':
            self.TOPIC_TEMPLATE = '告诉我这句话的主题，直接说主题不要解释：“{}”'
            self.SCORING_QUESTION_TEMPLTE = '“{}”\n请仔细阅读以上内容，判断句子是否是个有主题的疑问句，结果用 0～10 表示。直接提供得分不要解释。\n判断标准：有主语谓语宾语并且是疑问句得 10 分；缺少主谓宾扣分；陈述句直接得 0 分；不是疑问句直接得 0 分。直接提供得分不要解释。'  # noqa E501
            self.SCORING_RELAVANCE_TEMPLATE = '问题：“{}”\n材料：“{}”\n请仔细阅读以上内容，判断问题和材料的关联度，用0～10表示。判断标准：非常相关得 10 分；完全没关联得 0 分。直接提供得分不要解释。\n'  # noqa E501
            self.KEYWORDS_TEMPLATE = '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。搜索参数类型 string， 内容是短语或关键字，以空格分隔。\n你现在是{}交流群里的技术助手，用户问“{}”，你打算通过谷歌搜索查询相关资料，请提供用于搜索的关键字或短语，不要解释直接给出关键字或短语。'  # noqa E501
            self.SECURITY_TEMAPLTE = '判断以下句子是否涉及政治、辱骂、色情、恐暴、宗教、网络暴力、种族歧视等违禁内容，结果用 0～10 表示，不要解释直接给出得分。判断标准：涉其中任一问题直接得 10 分；完全不涉及得 0 分。直接给得分不要解释：“{}”'  # noqa E501
            self.PERPLESITY_TEMPLATE = '“question:{} answer:{}”\n阅读以上对话，answer 是否在表达自己不知道，回答越全面得分越少，用0～10表示，不要解释直接给出得分。\n判断标准：准确回答问题得 0 分；答案详尽得 1 分；知道部分答案但有不确定信息得 8 分；知道小部分答案但推荐求助其他人得 9 分；不知道任何答案直接推荐求助别人得 10 分。直接打分不要解释。'  # noqa E501
            self.SUMMARIZE_TEMPLATE = '{} \n 仔细阅读以上内容，总结得简短有力点'  # noqa E501
            # self.GENERATE_TEMPLATE = '材料：“{}”\n 问题：“{}” \n 请仔细阅读参考材料回答问题，材料可能和问题无关。如果材料和问题无关，尝试用你自己的理解来回答问题。如果无法确定答案，直接回答不知道。'  # noqa E501
            self.GENERATE_TEMPLATE = '材料：“{}”\n 问题：“{}” \n 请仔细阅读参考材料回答问题。'  # noqa E501
        else:
            self.TOPIC_TEMPLATE = 'Tell me the theme of this sentence, just state the theme without explanation: "{}"'  # noqa E501
            self.SCORING_QUESTION_TEMPLTE = '"{}"\nPlease read the content above carefully and judge whether the sentence is a thematic question. Rate it on a scale of 0-10. Only provide the score, no explanation.\nThe criteria are as follows: a sentence gets 10 points if it has a subject, predicate, object and is a question; points are deducted for missing subject, predicate or object; declarative sentences get 0 points; sentences that are not questions also get 0 points. Just give the score, no explanation.'  # noqa E501
            self.SCORING_RELAVANCE_TEMPLATE = 'Question: "{}", Background Information: "{}"\nPlease read the content above carefully and assess the relevance between the question and the material on a scale of 0-10. The scoring standard is as follows: extremely relevant gets 10 points; completely irrelevant gets 0 points. Only provide the score, no explanation needed.'  # noqa E501
            self.KEYWORDS_TEMPLATE = 'Google search is a general-purpose search engine that can be used to access the internet, look up encyclopedic knowledge, keep abreast of current affairs and more. Search parameters type: string, content consists of phrases or keywords separated by spaces.\nYou are now the assistant in the "{}" communication group. A user asked "{}", you plan to use Google search to find related information, please provide the keywords or phrases for the search, no explanation, just give the keywords or phrases.'  # noqa E501
            self.SECURITY_TEMAPLTE = 'Evaluate whether the following sentence involves prohibited content such as politics, insult, pornography, terror, religion, cyber violence, racial discrimination, etc., rate it on a scale of 0-10, do not explain, just give the score. The scoring standard is as follows: any violation directly gets 10 points; completely unrelated gets 0 points. Give the score, no explanation: "{}"'  # noqa E501
            self.PERPLESITY_TEMPLATE = 'Question: {} Answer: {}\nRead the dialogue above, does the answer express that they don\'t know? The more comprehensive the answer, the lower the score. Rate it on a scale of 0-10, no explanation, just give the score.\nThe scoring standard is as follows: an accurate answer to the question gets 0 points; a detailed answer gets 1 point; knowing some answers but having uncertain information gets 8 points; knowing a small part of the answer but recommends seeking help from others gets 9 points; not knowing any of the answers and directly recommending asking others for help gets 10 points. Just give the score, no explanation.'  # noqa E501
            self.SUMMARIZE_TEMPLATE = '"{}" \n Read the content above carefully, summarize it in a short and powerful way.'  # noqa E501
            self.GENERATE_TEMPLATE = 'Background Information: "{}"\n Question: "{}"\n Please read the reference material carefully and answer the question.'  # noqa E501
```



