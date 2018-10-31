# nlp
###起因
腾讯公开了训练好的语料库 [语料地址](https://ai.tencent.com/ailab/nlp/embedding.html)，我决定拿来用用；
介绍上显示还是很强大的，下载下来解压16G，txt 格式。直觉告诉我这是好东西，然后……没有然后了，不知道该怎么用，忙工作去。

-------------
3天后……
发现关键词 Word2vec,Gensim,一番搜索后，把环境配了一下 ubuntu 下 python3，pip3，gensim，Word2vec

```
sudo apt install -y python3-pip
sudo pip3 install --upgrade pip
sudo pip3 install gensim
sudo pip3 install jieba
```
**注意 python 版本问题**

测试代码
```
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(r'./model/Tencent_AILab_ChineseEmbedding.txt', binary=False,limit=50000)
print(model.most_similar(positive=['刘德华'], topn=10))
```
测试

`python3 using_txt.py`

输出结果：

`[('周润发', 0.889106273651123), ('梁朝伟', 0.8705732226371765), ('张学友', 0.8653705716133118), ('古天乐', 0.8521826267242432), ('张国荣', 0.8434552550315857), ('成龙', 0.8371275663375854), ('周星驰', 0.8290221691131592), ('郭富城', 0.8202318549156189), ('李连杰', 0.8052970170974731), ('梅艳芳', 0.8025732040405273)]`

到这里我们似乎已经成功了，其实还有几个疑点没有得到解决

###问题

1.速度有点慢，大概10秒才返回

2.可能出现目标词不在范围的问题

3.目前只加载了50000个热门词

4.如何训练自己的模型

5.如何在当前基础上继续增量训练

6.除了相似性还能干些什么

###解答
-------
前三个问题，速度慢主要是因为加载模型慢，模型太大，目前没找到好办法，期待大佬指点；如果完整加载，应该不会出现不在范围的问题了。

-----
后三个问题，让我们重头开始，做一下实践吧！

#####目标文本
选择大家耳熟能详的《人民的名义》

#####数据预处理
建议去掉里面的标点符号并替换为空格，各种缩进，空行，最好也能替换成空格
我比较懒，直接去除中文以外的所有字符，然后一行一行的读入数据处理

google 了一个现成的代码

```$xslt
import jieba
import re

class MySentence(object):
    def __init__(self,filename):
        self.filename = filename    

    def __iter__(self):
        lines = open(self.filename,'r',encoding='utf-8').readlines()
        for line in lines:
            #用正则剔除了除中文以外的所有字符
            ChineseSentence = ''.join(re.findall(r'[\u4e00-\u9fa5]', line))
            
            #jieba.cut生成的是生成器，这里要转换为列表
            wordlist = list(jieba.cut(ChineseSentence))
            
            yield wordlist
```

