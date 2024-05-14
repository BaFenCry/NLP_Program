import os
import shutil
import zipfile
import jieba
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import time
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from keras import models
from keras import layers
from keras.optimizers import *
from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback

class CustomCallback(Callback):
    def __init__(self,qt_self):
        super(CustomCallback, self).__init__()
        self.qt_self = qt_self

    def on_epoch_end(self, epoch, logs=None):
        # 在每个周期结束后获取数据并保存
        epoch_data = {
            'epoch': epoch+1,
            'train_loss': logs['loss'],
            'val_loss': logs['val_loss'],
            'train_accuracy': logs['accuracy'],
            'val_accuracy': logs['val_accuracy']
        }
        # 发射变量变化的信号
        self.qt_self.variable_changed.emit()
        self.qt_self.epoch_data=epoch_data


def read_text(path, text_list):
    '''
    path: 必选参数，文件夹路径
    text_list: 必选参数，文件夹 path 下的所有 .txt 文件名列表
    return: 返回值
        features 文本(特征)数据，以列表形式返回; 
        labels 分类标签，以列表形式返回
    '''
    
    features, labels = [], [] 
    for text in text_list:
        if text.split('.')[-1] == 'txt':
            try:
                with open(path + text, encoding='gbk') as fp:
                    features.append(fp.read())  # 特征
                    labels.append(path.split('/')[-2])  # 标签
            except Exception as erro:
                print('\n>>>发现错误, 正在输出错误信息...\n', erro)
                
    return features, labels


def merge_text(train_or_test, label_name):
    '''
    train_or_test: 必选参数，train 训练数据集 or test 测试数据集
    label_name: 必选参数，分类标签的名字
    return: 返回值
        merge_features 合并好的所有特征数据，以列表形式返回;
        merge_labels   合并好的所有分类标签数据，以列表形式返回
    '''
    
    print('\n>>>文本读取和合并程序已经启动, 请稍候...')
    
    merge_features, merge_labels = [], []  # 函数全局变量
    for name in label_name:
        path = 'text classification/'+ train_or_test +'/'+ name +'/'
        text_list = os.listdir(path)
        features, labels = read_text(path=path, text_list=text_list)  # 调用函数
        merge_features += features  # 特征
        merge_labels   += labels    # 标签
        
    # 可以自定义添加一些想要知道的信息
    print('\n>>>你正在处理的数据类型是...\n', train_or_test)
    print('\n>>>[', train_or_test ,']数据具体情况如下...')
    print('样本数量\t', len(merge_features), '\t类别名称\t', set(merge_labels))   
    print('\n>>>文本读取和合并工作已经处理完毕...\n')
    
    return merge_features, merge_labels

def get_text_classification(estimator, X, y, X_test, y_test,count):
    '''
    estimator: 分类器，必选参数
            X: 特征训练数据，必选参数
            y: 标签训练数据，必选参数
       X_test: 特征测试数据，必选参数
        y_tes: 标签测试数据，必选参数
       return: 返回值
           y_pred_model: 预测值
                  score: 准确率
                      t: 消耗的时间
                  matrix: 混淆矩阵
                  report: 分类评价函数
                       
    '''
    start = time.time()
    
    print('\n>>>算法正在启动，请稍候...')
    model = estimator
    
    print('\n>>>算法正在进行训练，请稍候...')
    model.fit(X, y)
    print(model)
    
    print('\n>>>算法正在进行预测，请稍候...')
    y_pred_model = model.predict(X_test)
    print(y_pred_model)
    
    print('\n>>>算法正在进行性能评估，请稍候...')
    score = metrics.accuracy_score(y_test, y_pred_model)
    matrix = metrics.confusion_matrix(y_test, y_pred_model)
    report = metrics.classification_report(y_test, y_pred_model)

    print('>>>准确率\n', score)
    print('\n>>>混淆矩阵\n', matrix)
    print('\n>>>召回率\n', report)
    print('>>>算法程序已经结束...')
    
    end = time.time()
    t = end - start
    print('\n>>>算法消耗时间为：', t, '秒\n')
    classifier = str(model).split('(')[0]
    
    return [report,y_pred_model , score, round(t, 2), matrix]

def classify_Bayes(self):
    # 读取合并后的CSV文件
    data = pd.read_csv(self.cur_file)
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
    # ### 中文文本分词
    # 训练集
    X_train_word = [jieba.cut(words) for words in X_train]
    X_train_cut = [' '.join(word) for word in X_train_word]
    # 测试集
    X_test_word = [jieba.cut(words) for words in X_test]
    X_test_cut = [' '.join(word) for word in X_test_word]
    # 停止词使用
    # 加载停止词语料
    stoplist = [word.strip() for word in open('data/stopword.txt',encoding='utf-8').readlines()]
    # ### 编码器处理文本标签
    le = LabelEncoder()
    y_train_le = le.fit_transform(y_train)
    y_test_le = le.fit_transform(y_test)

    # ### 文本数据转换成数据值数据矩阵
    count = CountVectorizer(stop_words=stoplist)
    count.fit(list(X_train_cut) + list(X_test_cut))
    X_train_count = count.transform(X_train_cut)
    X_test_count = count.transform(X_test_cut)
    X_train_count = X_train_count.toarray()
    X_test_count = X_test_count.toarray()

    # #### 多项式朴素贝叶斯
    mnb = MultinomialNB()
    result = get_text_classification(mnb, X_train_count, y_train_le, X_test_count, y_test_le,count)
    result.append(mnb)
    result.append(count)
    result.append(le)
    return result
    
    
def classify_SVM(self):
    # 读取合并后的CSV文件
    data = pd.read_csv(self.cur_file)
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
    # ### 中文文本分词
    # 训练集
    X_train_word = [jieba.cut(words) for words in X_train]
    X_train_cut = [' '.join(word) for word in X_train_word]
    # 测试集
    X_test_word = [jieba.cut(words) for words in X_test]
    X_test_cut = [' '.join(word) for word in X_test_word]
    # 停止词使用
    # 加载停止词语料
    stoplist = [word.strip() for word in open('data/stopword.txt',encoding='utf-8').readlines()]
    # ### 编码器处理文本标签
    le = LabelEncoder()
    y_train_le = le.fit_transform(y_train)
    y_test_le = le.fit_transform(y_test)

    # ### 文本数据转换成数据值数据矩阵
    count = CountVectorizer(stop_words=stoplist)
    count.fit(list(X_train_cut) + list(X_test_cut))
    X_train_count = count.transform(X_train_cut)
    X_test_count = count.transform(X_test_cut)
    X_train_count = X_train_count.toarray()
    X_test_count = X_test_count.toarray()

    # #### 多项式朴素贝叶斯
    svc = svm.SVC(kernel=self.cur_kernel)
    result = get_text_classification(svc, X_train_count, y_train_le, X_test_count, y_test_le,count)
    result.append(svc)
    result.append(count)
    result.append(le)
    return result

def classify_FNN(self):
    if self.cur_optimizer=="SGD":
        optimizer=SGD(lr=float(self.cur_rate))
    elif self.cur_optimizer=="Adam":
        optimizer=Adam(lr=float(self.cur_rate))
    elif self.cur_optimizer=="RMSprop":
        optimizer=RMSprop(lr=float(self.cur_rate))

    # 读取合并后的CSV文件
    data = pd.read_csv(self.cur_file)
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
    # 训练集
    X_train_word = [jieba.cut(words) for words in X_train]
    X_train_cut = [' '.join(word) for word in X_train_word]
    # 测试集
    X_test_word = [jieba.cut(words) for words in X_test]
    X_test_cut = [' '.join(word) for word in X_test_word]
    # 加载停止词语料
    stoplist = [word.strip() for word in open('data/stopword.txt',encoding='utf-8').readlines()]
    # ### 编码器处理文本标签
    le = LabelEncoder()
    y_train_le = le.fit_transform(y_train)
    y_test_le = le.fit_transform(y_test)
    # ### 文本数据转换成数据值数据矩阵
    count = CountVectorizer(stop_words=stoplist)
    count.fit(list(X_train_cut) + list(X_test_cut))
    X_train_count = count.transform(X_train_cut)
    X_test_count = count.transform(X_test_cut)

    X_train_count = X_train_count.toarray()
    X_test_count = X_test_count.toarray()

    # 深度学习算法———多分类前馈神经网络
    start = time.time()
    # --------------------------------
    # np.random.seed(0)     # 设置随机数种子
    feature_num = X_train_count.shape[1]     # 设置所希望的特征数量
    # ---------------------------------
    # 独热编码目标向量来创建目标矩阵
    y_train_cate = to_categorical(y_train_le)
    y_test_cate = to_categorical(y_test_le)
    print(y_train_cate)
    # ----------------------------------------------------
    # 创建神经网络
    network = models.Sequential() 
    # ----------------------------------------------------
    # 添加神经连接层
    # 第一层必须有并且一定是 [输入层], 必选
    network.add(layers.Dense(     # 添加带有 relu 激活函数的全连接层
                            units=128, 
                            activation='relu', 
                            input_shape=(feature_num, )
                            ))

    # 介于第一层和最后一层之间的称为 [隐藏层]，可选
    network.add(layers.Dense(     # 添加带有 relu 激活函数的全连接层
                            units=128, 
                            activation=self.cur_kernel
                            ))
    network.add(layers.Dropout(0.8))
    # 最后一层必须有并且一定是 [输出层], 必选                         
    network.add(layers.Dense(     # 添加带有 softmax 激活函数的全连接层
                            units=4,
                            activation='sigmoid'
                            ))
    # -----------------------------------------------------
    # 编译神经网络
    network.compile(loss='categorical_crossentropy',  # 分类交叉熵损失函数    
                    optimizer=optimizer,  
                    metrics=['accuracy']   ,        # 准确率度量
                    )
    # -----------------------------------------------------
    # 训练神经网络
    history = network.fit(X_train_count,     # 训练集特征
                y_train_cate,        # 训练集标签
                epochs=int(self.cur_epoch)  ,      # 迭代次数
                batch_size=int(self.cur_batch),    # 每个批量的观测数  可做优化
                validation_data=(X_test_count, y_test_cate),  # 验证测试集数据
                callbacks=[CustomCallback(self)]          
                )
    network.summary()

    #  模型预测
    y_pred_keras = network.predict(X_test_count)

    # y_pred_keras[:20]
    # -----------------------------------------------------
    #  性能评估
    print('>>>多分类前馈神经网络性能评估如下...\n')
    score = network.evaluate(X_test_count,
                            y_test_cate,
                            batch_size=int(self.cur_batch))
    print('\n>>>评分\n', score)
    end = time.time()

    # 将模型的预测结果转换成类别标签
    y_pred_labels = np.argmax(y_pred_keras, axis=1)

    # 生成性能评估报告
    report = metrics.classification_report(y_test_le, y_pred_labels)

    return [report,network,count,le,score]


