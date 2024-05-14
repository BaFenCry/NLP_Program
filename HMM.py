import numpy as np
from pyhanlp import HanLP
import os

def train(self):

    # 定义一个状态映射字典。方便我们定位状态在列表中对应位置
    status2num={'B':0,'M':1,'E':2,'S':3}

    # 定义状态转移矩阵。总共4个状态，所以4x4
    A=np.zeros((4,4))
    B=np.zeros((4,65536))

    # 初始状态，每一个句子的开头只有4中状态（词性）
    PI=np.zeros(4)

    with open(self.cur_file,encoding='utf-8') as file:
        for line in file.readlines():
            wordStatus=[]#用于保存该行所有单词的状态
            words=line.strip().split() #除去前后空格，然后依照中间空格切分为单词

            for i,word in enumerate(words):

                # 根据长度判断状态
                if len(word)==1:
                    status='S'# 保存每一个单词状态
                    # 使用ord找到该字对应编码
                    # 更新B矩阵
                    # B代表了每一个状态到对应观测结果的可能性
                    # 先统计频数
                    code=ord(word)
                    B[status2num[status[0]]][code]+=1

                else:
                    # 当长度为2，M*0。这样可以一起更新
                    status='B'+(len(word)-2)*'M'+'E'
                    # 使用ord找到该字对应编码
                    # 更新B矩阵
                    # B代表了每一个状态到对应观测结果的可能性
                    # 先统计频数
                    for s in range(len(word)):
                        code=ord(word[s])
                        B[status2num[status[s]]][code]+=1

                # i==0意味着这是句首。我们需要更新PI中每种状态出现次数
                if i==0:
                    # status[0]表示这行第一个状态
                    # status2num将其映射到list对应位置
                    PI[status2num[status[0]]]+=1

                # 使用extend，将status中每一个元素家在列表之中。而不是append直接将整个status放在后面
                wordStatus.extend(status)

            # 遍历完了一行，然后更新矩阵A
            # A代表的是前一个状态到后一个状态的概率
            # 我们先统计频数
            for i in range(1,len(wordStatus)):
                # wordStatus获得状态，使用status2num来映射到正确位置
                A[status2num[wordStatus[i-1]]][status2num[wordStatus[i]]]+=1


    # 计算PI向量
    total=sum(PI)
    for i in range(len(PI)):
        if PI[i]==0:
            PI[i]=-3.14e+100
        else:
            # 别忘了去取对数
            PI[i]=np.log(PI[i]/total)

    for i in range(len(A)):
        total=sum(A[i])
        for j in range(len(A[i])):
            if A[i][j]==0:
                A[i][j]=-3.14e+100
            else:
                A[i][j]=np.log(A[i][j]/total)
    # 更新B矩阵
    # B矩阵中，每一行只和为1
    # 即某一个状态到所有观测结果只和为1
    # 最后我们取对数
    for i in range(len(B)):
        total=sum(B[i])
        for j in range(len(B[i])):
            if B[i][j]==0:
                B[i][j]=-3.14e+100
            else:
                B[i][j]=np.log(B[i][j]/total)
    print("序列标注训练。。。。。")
    # 返回三个参数
    return (PI,A,B)


def word_partition(HMM_parameter, article):
    '''
    使用维特比算法进行预测（即得到路径中每一个最有可能的状态）
    :param HMM_parameter: PI,A,B隐马尔可夫模型三要素
    :param article: 需要分词的文章,以数组的形式传入，每一个元素是一行
    :return: 分词后的文章以及相应的状态标签序列
    '''
    PI, A, B = HMM_parameter
    article_partition = []  # 分词之后的文章
    tag_sequence = []  # 相应的状态标签序列

    for line in article:
        delta = [[0 for _ in range(4)] for _ in range(len(line))]  # 定义delta
        psi = [[0 for _ in range(4)] for _ in range(len(line))]  # 定义psi

        for t in range(len(line)):
            if t == 0:
                psi[t][:] = [0, 0, 0, 0]
                for i in range(4):
                    delta[t][i] = PI[i] + B[i][ord(line[t])]

            else:
                for i in range(4):
                    temp = [delta[t-1][j] + A[j][i] for j in range(4)]
                    delta[t][i] = max(temp) + B[i][ord(line[t])]
                    psi[t][i] = temp.index(max(temp))

        status = []  # 保存最优状态链
        tag_line = []  # 保存每个字对应的状态标签

        It = delta[-1].index(max(delta[-1]))
        status.append(It)
        for t in range(len(delta)-2, -1, -1):
            It = psi[t+1][status[0]]
            status.insert(0, It)

        for t in range(len(line)):
            if status[t] == 0:
                tag = 'B'
            elif status[t] == 1:
                tag = 'M'
            elif status[t] == 2:
                tag = 'E'
            else:
                tag = 'S'
            tag_line.append(tag)

        line_partition = ''
        for t in range(len(line)):
            line_partition += line[t]
            if (status[t] == 2 or status[t] == 3) and t != len(line)-1:
                line_partition += ' '

        article_partition.append(line_partition)
        tag_sequence.append(tag_line)

    return article_partition, tag_sequence


def loadArticle(fileName):
    '''
    读取测试文章
    :param fileName: 文件名
    :return: 处理之后的文章
    '''
    # 我们需要将其空格去掉
    with open(fileName,encoding='utf-8') as file:
        # 按行读取
        test_article=[]
        for line in file.readlines():
            # 去除空格，以及换行符
            line=line.strip()
            test_article.append(line)
    return test_article




def generate_gold_standard(test_file, gold_standard_file):
    # 读取测试文件
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = f.readlines()

    # 初始化标准答案列表
    gold_standard = []

    # 对每一行测试数据进行分词，并将分词结果写入标准答案列表
    for line in test_data:
        segmented_line = HanLP.segment(line.strip())
        gold_standard.append([term.word for term in segmented_line])

    # 将标准答案写入文件
    with open(gold_standard_file, 'w', encoding='utf-8') as f:
        for line in gold_standard:
            f.write(' '.join(line) + '\n')

def write_predicted_result(article_partition, predicted_file):
    with open(predicted_file, 'w', encoding='utf-8') as f:
        for line in article_partition:
            f.write(line + '\n')


def HMM_print(article_partition,tag_sequence):
    predicted_file = 'predicted.txt'
    write_predicted_result(article_partition, predicted_file)


    print("分词结果及相应的状态标签序列：")
    for i in range(len(article_partition)):
        print("分词结果：", article_partition[i])
        print("状态标签序列：", ''.join(tag_sequence[i]))
        print()

    test_file = 'test.txt'
    gold_standard_file = 'gold_standard.txt'
    # 生成标准答案文件
    generate_gold_standard(test_file, gold_standard_file)


