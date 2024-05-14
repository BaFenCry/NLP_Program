from HMM import *
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report



def convert_to_BMES(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as input_file, open(output_file, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            line = line.strip()  # 去除换行符等空白字符
            words = line.split()  # 按空格分割单词
            for word in words:
                if len(word) == 1:
                    output_file.write("S")
                else:
                    output_file.write("B" + "M" * (len(word) - 2) + "E")
            output_file.write("\n")  # 每处理完一行，换行

# # 使用示例
# convert_to_BMES("gold_standard.txt", "BMES_tags.txt")
# convert_to_BMES("predicted.txt", "test-xulie.txt")
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]

def calculate_accuracy(seq1, seq2):
    total_tokens = 0
    correct_tokens = 0

    for token1, token2 in zip(seq1, seq2):
        total_tokens += len(token1)
        for t1, t2 in zip(token1, token2):
            if t1 == t2:
                correct_tokens += 1

    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    return accuracy

def calculate_precision_recall_f1(seq1, seq2):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for token1, token2 in zip(seq1, seq2):
        for t1, t2 in zip(token1, token2):
            if t1 == 'B' and t2 == 'B':
                true_positives += 1
            elif t1 == 'B' and t2 != 'B':
                false_positives += 1
            elif t1 != 'B' and t2 == 'B':
                false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def convert_to_onehot(seq):
    # 创建一个LabelBinarizer对象，它会返回一个二进制矩阵
    # classes参数指定了所有的类别，以便可以正确地进行独热编码
    lb = LabelBinarizer()
    lb.fit(['B', 'M', 'E', 'S'])
    onehot = lb.transform(seq)
    return onehot

# # 读取样本集序列和标准答案集序列
# test_seq = read_file("test-xulie.txt")
# standard_seq = read_file("BMES_tags.txt")


def evaluate_HMM(test_seq, standard_seq):
    
    # 将标签序列转换为独热编码
    y_true = convert_to_onehot(standard_seq)
    y_pred = convert_to_onehot(test_seq)
    # 计算准确率
    accuracy = calculate_accuracy(test_seq, standard_seq)

    # 计算精确率、召回率和 F1 值
    precision, recall, f1_score = calculate_precision_recall_f1(test_seq, standard_seq)

    # # 使用sklearn的classification_report生成分类报告
    # report = classification_report(y_true, y_pred, target_names=['B', 'M', 'E', 'S'])

    #输出评估结果
    print("评估结果：")
    print("准确率 (Accuracy):", accuracy)
    print("精确率 (Precision):", precision)
    print("召回率 (Recall):", recall)
    print("F1 值 (F1 Score):", f1_score)

# # 评估模型
# evaluate_HMM(test_seq, standard_seq)