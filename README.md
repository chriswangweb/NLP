# nlp
### 起因
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
** 注意 python 版本问题 **

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

### 问题

1.速度有点慢，大概10秒才返回

2.可能出现目标词不在范围的问题

3.目前只加载了50000个热门词

4.如何训练自己的模型

5.如何在当前基础上继续增量训练

6.除了相似性还能干些什么

### 解答
-------
前三个问题，速度慢主要是因为加载模型慢，模型太大，目前没找到好办法，期待大佬指点；如果完整加载，应该不会出现不在范围的问题了。

-----
后三个问题，让我们重头开始，做一下实践吧！

##### 1.确定目标文本
选择大家耳熟能详的《人民的名义》

##### 2.数据预处理
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

#### 3.训练自己的模型

```$xslt
MySentences = MySentence(filename=r'./人民的名义.txt')

#训练模型
model = Word2Vec(sentences=MySentences,
                 size=200,
                 window=10,
                 min_count=10,
                 workers=4)

#保存模型                
model.wv.save_word2vec_format(fname=r'./model/renmin_word2vec_binary.bin',binary=True)
```
然后你会看到这样的结果，表示训练成功
```$xslt
chris@xxx-s1:~$ python3 generate_model.py 
Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
Loading model cost 1.102 seconds.
Prefix dict has been built succesfully.
16.410267114639282
```

#### 4.使用刚刚训练的模型
基本操作了，代码如下

```$xslt
from gensim.models import KeyedVectors
#导入模型
model = KeyedVectors.load_word2vec_format(r'./model/santi_word2vec_binary.bin', binary=True)
print(model.most_similar(positive=['高育良'], topn=10))
```
神马，返回结果说这个词都不在范围！！应该是人名识别有问题，决定使用jieba.add_word处理

搜集人名如下

```$xslt
侯亮平
陆亦可
沙瑞金
李达康
祁同伟
高育良
高小琴
吴惠芬
刘新建
陈岩石
季昌明
赵瑞龙
郑西坡
钟小艾
赵东来
蔡成功
欧阳菁
田国富
丁义珍
赵徳汉
易学习
王馥真
梁璐
程度
陈海
孙连城
郑胜利
林华华
周正
张宝宝
陈群芳
吴心仪
肖钢玉
王大路
吕梁
王国风
陈清泉
王文革
```
重新训练，测试
得到如下结果
```$xslt
[('沙瑞金', 0.9998887777328491), ('学生', 0.9998694062232971), ('向', 0.9998139142990112), ('祁同伟', 0.9998072385787964), ('汇报', 0.9998005628585815), ('看', 0.999794602394104), ('李达康', 0.999789297580719), ('易学习', 0.9997716546058655), ('意见', 0.9997698068618774), ('苦笑', 0.9997631311416626)]
[('你', -0.9805782437324524), ('我', -0.9833072423934937), ('大风', -0.9913561344146729), ('厂', -0.9914196729660034), ('的', -0.9919933080673218), ('山水', -0.9923231601715088), ('集团', -0.9933937788009644), ('说', -0.9943312406539917), ('持股', -0.9952439665794373), ('啊', -0.9957468509674072)]
```
看起来还是比较靠谱的，原来育良书记最亲近的是沙瑞金，最爱说汇报，经常苦笑，学生是祁同伟，经常提大风厂，山水集团，还有可能在其中持股