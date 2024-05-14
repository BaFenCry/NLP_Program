import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import QTimer,Qt
import joblib
import jieba
import torch
import torchkeras
from torchcrf import CRF
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from classification import classify_Bayes,classify_SVM,classify_FNN
# from NER import ner_bilstm_crf,predict,get_valid_nertag
from NER_2 import NER_2
from cluster import cluster_K_means,predict_text,chinese_word_cut
import threading
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import functools
from keras import layers, models
from HMM import train,word_partition
from evaluate import *
from NER_2 import *

def read_file_to_string(file_name):
    """
    读取文件内容并返回为一个字符串。
    
    参数:
    file_name -- 要读取的文件的名称
    
    返回:
    文件内容的字符串
    """
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"文件 {file_name} 未找到。")
        return None
    except Exception as e:
        print(f"读取文件 {file_name} 时发生错误: {e}")
        return None

def transform_number(n):
    if n == 2:
        return 1
    elif n == 3:
        return 2
    elif n == 0:
        return 3
    elif n == 1:
        return 4
    else:
        # 如果输入的数字不在指定的范围内，可以选择返回None或者抛出异常
        return None  

# Worker 类，用于在单独的线程中运行任务
class Worker(QObject):
    # 任务完成的信号，带有结果类型
    finished = pyqtSignal(object)

    def __init__(self, classify_function):
        super(Worker, self).__init__()
        self.classify_function = classify_function

    # 在单独线程中运行任务的方法
    def run_task(self):
        # 调用指定的函数并发送结果信号
        result = self.classify_function()
        self.finished.emit(result)


class MyWindow(QWidget):
    variable_changed = pyqtSignal()
    def __init__(self):
        super().__init__()
        # 创建一个空字典
        self.map = {}
        # 添加键值对
        self.map['序列标注'] = 'sequence_labeling'
        self.map['命名实体识别'] = 'NER'
        self.map['文本聚类'] = 'clustering'
        self.map['文本分类'] = 'classification'
        self.model=None
        self.epoch_data=[]
        flag=1
        # 定义一个变量变化的信号
        self.variable_changed.connect(self.draw)
        self.evaluating_ndicator=None
        self.initUI()
        self.select_model.setCurrentText("隐马尔可夫")
        self.learning_rate_lineEdit.setText("当前选中模型无法使用该参数")
        self.batch_lineEdit.setText("当前选中模型无法使用该参数")
        self.set_epoch_lineEdit.setText("当前选中模型无法使用该参数")
        self.set_K_lineEdit.setText("当前选中模型无法使用该参数")

    def create_first_vbox_layout(self):
        # 创建垂直布局
        self.vbox_layout_1 = QVBoxLayout()
        # 输入文本
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText('这里用来进行输入文本') 
        self.hbox_layout_1 = QHBoxLayout()
        self.hbox_layout_1.addWidget(self.input_edit)
        
        # 序列标注
        self.sequence_labeling = QTextEdit()
        self.sequence_labeling.setPlaceholderText('这里用来显示进行序列标注结果')
        self.select_model_button_2=QPushButton("选择模型")
        self.select_model_button_2.clicked.connect(self.select_model_2)
        self.select_model_label_2=QLabel(" 未选择模型")
        self.button_2 = QPushButton('开始')
        self.button_2.clicked.connect(self.use_model_2)
        self.vbox_2=QVBoxLayout()
        self.vbox_2.addWidget(self.select_model_button_2)
        self.vbox_2.addWidget(self.select_model_label_2)
        self.vbox_2.addWidget(self.button_2)
        self.hbox_layout_2 = QHBoxLayout()  
        self.hbox_layout_2.addWidget(self.sequence_labeling)
        self.hbox_layout_2.addLayout(self.vbox_2)

        # 命名实体识别
        self.NER = QTextEdit()
        self.NER.setPlaceholderText('这里用来显示进行命名实体识别结果') 
        self.select_model_button_3=QPushButton("选择模型")
        self.select_model_button_3.clicked.connect(self.select_model_3)
        self.select_model_label_3=QLabel(" 未选择模型")
        self.button_3 = QPushButton('开始')
        self.button_3.clicked.connect(self.use_model_3)
        self.vbox_3=QVBoxLayout()
        self.vbox_3.addWidget(self.select_model_button_3)
        self.vbox_3.addWidget(self.select_model_label_3)
        self.vbox_3.addWidget(self.button_3)
        self.hbox_layout_3 = QHBoxLayout()  
        self.hbox_layout_3.addWidget(self.NER)
        self.hbox_layout_3.addLayout(self.vbox_3)

        # 聚类
        self.cluster = QTextEdit()
        self.cluster.setPlaceholderText('这里用来显示进行聚类结果') 
        self.select_model_button_4=QPushButton("选择模型")
        self.select_model_button_4.clicked.connect(self.select_model_4)
        self.select_model_label_4=QLabel(" 未选择模型")
        self.button_4 = QPushButton('开始')
        self.button_4.clicked.connect(self.use_model_4)
        self.vbox_4=QVBoxLayout()
        self.vbox_4.addWidget(self.select_model_button_4)
        self.vbox_4.addWidget(self.select_model_label_4)
        self.vbox_4.addWidget(self.button_4)
        self.hbox_layout_4 = QHBoxLayout()  
        self.hbox_layout_4.addWidget(self.cluster)
        self.hbox_layout_4.addLayout(self.vbox_4)

        # 分类
        self.classification = QTextEdit()
        self.classification.setPlaceholderText('这里用来显示进行分类结果') 
        self.select_model_button_5=QPushButton("选择模型")
        self.select_model_button_5.clicked.connect(self.select_model_5)
        self.select_model_label_5=QLabel(" 未选择模型")
        self.button_5 = QPushButton('开始')
        self.button_5.clicked.connect(self.use_model_5)
        self.vbox_5=QVBoxLayout()
        self.vbox_5.addWidget(self.select_model_button_5)
        self.vbox_5.addWidget(self.select_model_label_5)
        self.vbox_5.addWidget(self.button_5)
        self.hbox_layout_5 = QHBoxLayout()  
        self.hbox_layout_5.addWidget(self.classification)
        self.hbox_layout_5.addLayout(self.vbox_5)

        # 添加水平控件到表单控件
        self.vbox_layout_1.addLayout(self.hbox_layout_1)
        self.vbox_layout_1.addLayout(self.hbox_layout_2)
        self.vbox_layout_1.addLayout(self.hbox_layout_3)
        self.vbox_layout_1.addLayout(self.hbox_layout_4)
        self.vbox_layout_1.addLayout(self.hbox_layout_5)


    def create_second_vbox_layout(self):
        # 第二个垂直布局
        self.vbox_layout_2 = QVBoxLayout()
        #一个水平布局
        self.hbox_parameter_selection=QHBoxLayout()
        ##############选择文件的垂直布局
        self.vbox_selcet_file=QVBoxLayout()
        self.select_file_label = QLabel(" 未选数据集", self)
        self.select_file_button = QPushButton('选择数据集', self)
        self.select_file_button.clicked.connect(self.browse_file)
        self.vbox_selcet_file.addWidget(self.select_file_label)
        self.vbox_selcet_file.addWidget(self.select_file_button)
        self.hbox_parameter_selection.addLayout(self.vbox_selcet_file)

        #############模型参数的垂直布局
        self.vbox_model_parameter=QVBoxLayout()

        #一个关联label和任务选择的水平布局
        self.hbox_task_select=QHBoxLayout()
        self.task_label=QLabel("任务选择:",self)
        self.select_task=QComboBox(self)
        self.select_task.addItems(["序列标注","命名实体识别","文本聚类","文本分类"])
        self.hbox_task_select.addWidget(self.task_label)
        self.hbox_task_select.addWidget(self.select_task)

        #一个关联label和model的水平布局
        self.hbox_model=QHBoxLayout()
        self.model_class_label=QLabel("选择模型:",self)
        self.select_model=QComboBox(self)
        self.add_options(self.select_model,"model")
        self.hbox_model.addWidget(self.model_class_label)
        self.hbox_model.addWidget(self.select_model)
        
        #一个关联label和K的水平布局
        self.hbox_K=QHBoxLayout()
        self.K_label=QLabel("簇数目 :",self)
        self.set_K_lineEdit=QLineEdit()
        self.hbox_K.addWidget(self.K_label)
        self.hbox_K.addWidget(self.set_K_lineEdit)

        #一个关联label和epoch的水平布局
        self.hbox_epoch=QHBoxLayout()
        self.epoch_label=QLabel("训练周期:",self)
        self.set_epoch_lineEdit=QLineEdit()
        self.hbox_epoch.addWidget(self.epoch_label)
        self.hbox_epoch.addWidget(self.set_epoch_lineEdit)

        #一个关联label和核函数的水平布局
        self.hbox_kernel=QHBoxLayout()
        self.kernel_label=QLabel("核(激活)函数:",self)
        self.select_kernel=QComboBox(self)
        self.add_options(self.select_kernel,"kernel")
        self.hbox_kernel.addWidget(self.kernel_label)
        self.hbox_kernel.addWidget(self.select_kernel)

        #一个关联label和优化器的水平布局
        self.hbox_optimizer=QHBoxLayout()
        self.optimizer_label=QLabel("优化器  :",self)
        self.select_optimizer=QComboBox()
        self.hbox_optimizer.addWidget(self.optimizer_label)
        self.hbox_optimizer.addWidget(self.select_optimizer)

        #一个关联label和学习率的水平布局
        self.hbox_learning_rate=QHBoxLayout()
        self.learning_rate_label=QLabel("学习率  :",self)
        self.learning_rate_lineEdit=QLineEdit()
        self.hbox_learning_rate.addWidget(self.learning_rate_label)
        self.hbox_learning_rate.addWidget(self.learning_rate_lineEdit)

        #一个关联label和批次大小的水平布局
        self.hbox_batch=QHBoxLayout()
        self.batch_label=QLabel("批次大小:",self)
        self.batch_lineEdit=QLineEdit()
        self.hbox_batch.addWidget(self.batch_label)
        self.hbox_batch.addWidget(self.batch_lineEdit)

        #将水平布局放入垂直布局
        self.vbox_model_parameter.addLayout(self.hbox_task_select)
        self.vbox_model_parameter.addLayout(self.hbox_model)
        self.vbox_model_parameter.addLayout(self.hbox_K)
        self.vbox_model_parameter.addLayout(self.hbox_epoch)
        self.vbox_model_parameter.addLayout(self.hbox_kernel)
        self.vbox_model_parameter.addLayout(self.hbox_optimizer)
        self.vbox_model_parameter.addLayout(self.hbox_learning_rate)
        self.vbox_model_parameter.addLayout(self.hbox_batch)
        
        #垂直布局放入水平布局
        self.hbox_parameter_selection.addLayout(self.vbox_model_parameter)
        
        #############模型训练的垂直布局
        self.vbox_training=QVBoxLayout()
        self.start_train_button=QPushButton("开始训练",self)
        self.start_train_button.clicked.connect(self.start_train_thread)
        self.vbox_training.addWidget(self.start_train_button)
        # self.vbox_training.addWidget(QPushButton("停止训练",self))
        self.save_model_button=QPushButton("保存模型",self)
        self.save_model_button.clicked.connect(self.save_model)
        self.vbox_training.addWidget(self.save_model_button)

        self.hbox_parameter_selection.addLayout(self.vbox_training)

        # 水平布局放入垂直布局
        self.vbox_layout_2.addLayout(self.hbox_parameter_selection)
    
        # #########################页面选项卡
        self.stacked_widget = QStackedWidget()
        # self.stacked_widget.addWidget(self.accuracy_chart)

        # 创建曲线图
        self.accuracy_chart = QChart()
        self.accuracy_chart.setTitle("准确率")
        # 创建数据系列
        self.train_accuracy_series = QLineSeries()
        self.train_accuracy_series.setName("训练集准确率")
        self.val_accuracy_series = QLineSeries()
        self.val_accuracy_series.setName("验证集准确率")
        # 添加数据
        self.train_accuracy_series.setPointsVisible(True)
        # self.train_accuracy_series.setPointLabelsVisible(True)
        # self.train_accuracy_series.setPointLabelsFormat("@yPoint")
        self.val_accuracy_series.setPointsVisible(True)
        # self.val_accuracy_series.setPointLabelsVisible(True)
        # self.val_accuracy_series.setPointLabelsFormat("@yPoint")
        # 显示数据
        self.accuracy_chart.addSeries(self.train_accuracy_series)
        self.accuracy_chart.setVisible(True)
        self.accuracy_chart.addSeries(self.val_accuracy_series)
        self.accuracy_chart.setVisible(True)

        # 创建横坐标轴
        self.axis_x = QValueAxis()
        self.axis_x.setLabelFormat("%.1f")  # 设置标签格式
        self.axis_x.setTitleText("Epoch")  # 设置标题
        self.accuracy_chart.addAxis(self.axis_x, Qt.AlignBottom)  # 添加到图表并设置位置
        # 创建纵坐标轴
        self.axis_y_loss = QValueAxis()
        self.axis_y_loss.setLabelFormat("%.1f")  # 设置标签格式
        self.axis_y_loss.setTitleText("Loss")  # 设置标题
        self.accuracy_chart.addAxis(self.axis_y_loss, Qt.AlignLeft)  # 添加到图表并设置位置
        # 将数据系列和轴关联
        self.train_accuracy_series.attachAxis(self.axis_x)
        self.train_accuracy_series.attachAxis(self.axis_y_loss)
        self.val_accuracy_series.attachAxis(self.axis_x)
        self.val_accuracy_series.attachAxis(self.axis_y_loss)
        # 创建图表视图
        self.accuracy_chart_view = QChartView(self.accuracy_chart)
        self.accuracy_chart_view.setRenderHint(QPainter.Antialiasing)

        self.stacked_widget.addWidget(self.accuracy_chart_view)

        ####################################################################################################
        # 创建损失值页面
        self.loss_chart = QChart()
        self.loss_chart.setTitle("损失值")
        # 创建数据系列
        self.train_loss_series = QLineSeries()
        self.train_loss_series.setName("训练集损失值")
        self.val_loss_series = QLineSeries()
        self.val_loss_series.setName("验证集损失值")
        # 添加数据
        self.train_loss_series.setPointsVisible(True)
        self.val_loss_series.setPointsVisible(True)
        # 显示数据
        self.loss_chart.addSeries(self.train_loss_series)
        self.loss_chart.setVisible(True)
        self.loss_chart.addSeries(self.val_loss_series)
        self.loss_chart.setVisible(True)

        # 创建横坐标轴
        self.axis_x_loss = QValueAxis()
        self.axis_x_loss.setLabelFormat("%.1f")  # 设置标签格式
        self.axis_x_loss.setTitleText("Epoch")  # 设置标题
        self.loss_chart.addAxis(self.axis_x_loss, Qt.AlignBottom)  # 添加到图表并设置位置
        # 创建纵坐标轴
        self.axis_y_loss = QValueAxis()
        self.axis_y_loss.setLabelFormat("%.1f")  # 设置标签格式
        self.axis_y_loss.setTitleText("Loss")  # 设置标题
        self.loss_chart.addAxis(self.axis_y_loss, Qt.AlignLeft)  # 添加到图表并设置位置
        # 将数据系列和轴关联
        self.train_loss_series.attachAxis(self.axis_x_loss)
        self.train_loss_series.attachAxis(self.axis_y_loss)
        self.val_loss_series.attachAxis(self.axis_x_loss)
        self.val_loss_series.attachAxis(self.axis_y_loss)
        # 创建图表视图
        self.loss_chart_view = QChartView(self.loss_chart)
        self.loss_chart_view.setRenderHint(QPainter.Antialiasing)
        self.stacked_widget.addWidget(self.loss_chart_view)
        ####################################################################################################

        # 创建评价指标页面
        self.evaluation_text_edit = QTextEdit()
        self.stacked_widget.addWidget(self.evaluation_text_edit)
        
        # 创建工具栏
        self.toolbar = QToolBar()
        # 创建工具栏按钮
        self.create_buttons(self.toolbar)

        self.vbox_layout_2.addWidget(self.toolbar)
        self.vbox_layout_2.addWidget(self.stacked_widget)


        ##################训练进度条
        self.hbox_progress=QHBoxLayout()
        self.progress_label=QLabel("训练进度:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.hbox_progress.addWidget(self.progress_label)
        self.hbox_progress.addWidget(self.progress_bar)
        self.vbox_layout_2.addLayout(self.hbox_progress)

        ###############退出按钮
        # 添加退出按钮
        self.exit_button = QPushButton("退出")
        self.exit_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # 设置按钮大小策略
        # 将退出按钮的点击事件连接到关闭窗口的操作
        self.exit_button.clicked.connect(self.close)
        self.vbox_layout_2.addWidget(self.exit_button)

    def create_chart(self,name):
        # 创建一个简单的图表
        chart = QChart()
        chart.setTitle(name)
        chart.legend().hide()
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        return chart_view

    def create_buttons(self, toolbar):
        # 创建工具栏按钮
        buttons = ["准确率", "损失值", "评价指标"]
        for index, text in enumerate(buttons):
            button = QToolButton(toolbar)
            button.setText(text)
            # 使用 functools.partial 创建带参数的函数
            button.clicked.connect(functools.partial(self.change_page, index))
            toolbar.addWidget(button)

    def change_page(self, index):
        # 切换页面
        self.stacked_widget.setCurrentIndex(index)

    def browse_file(self):
        options = QFileDialog.Options()
        self.cur_task=self.select_task.currentText()
        if(self.cur_task=="命名实体识别"):
            file_names, _ = QFileDialog.getOpenFileNames(self, "选择文件", "", "所有文件 (*);;文本文件 (*.txt);;CSV 文件 (*.csv)", options=options)
            if file_names :
                selected_files = ', '.join(os.path.basename(file) for file in file_names)
                self.select_file_label.setText(f"已选择: {selected_files}")
                self.select_file_name = file_names  # 将所有选中的文件路径保存在 
                print(self.select_file_name)
        else:
            file_name, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "所有文件 (*);;文本文件 (*.txt);;CSV 文件 (*.csv)", options=options)
            if file_name :
                if(self.check_file_format(file_name)):
                    self.select_file_label.setText(f"已选择: {os.path.basename(file_name)}")
                    self.select_file_name=file_name

    

    #判断文件格式
    def check_file_format(self, file_path):
        _, extension = os.path.splitext(file_path)
        if extension.lower() not in ['.txt', '.csv']:  # 检查是否为支持的文件类型
            QMessageBox.warning(self, "文件类型错误", f"不支持的文件类型: {extension}\n请重新选择一个.txt或.csv文件。")
            return False
        return True
    
    #判断模型格式
    def check_model_format(self, file_path):
        _, extension = os.path.splitext(file_path)
        if extension.lower() not in ['.pkl','.h5','.tar']:  # 检查是否为支持的文件类型
            QMessageBox.warning(self, "文件类型错误", f"不支持的文件类型: {extension}\n请重新选择一个.pkl文件。")
            return False
        return True
    
    def add_options(self,now,type):
        if(type=="model"):
            now.addItems(["隐马尔可夫"])
        # elif(type=="kernel"):
        #     selected_text = self.select_model.currentText() 
        #     if(selected_text=="SVM"):
        #         now.addItems(["linear","sigmoid","poly"])
        #     elif(selected_text!="贝叶斯"):
        #         now.addItems(["relu","sigmoid","tanh"])
    
    def control_widget(self):
        self.select_kernel.clear()
        self.select_optimizer.clear()
        selected_text = self.select_model.currentText() 
        if(selected_text=="SVM"):
            self.select_kernel.addItems(["linear","sigmoid","poly"])
            self.learning_rate_lineEdit.setText("当前选中模型无法使用该参数")
            self.batch_lineEdit.setText("当前选中模型无法使用该参数")
            self.set_epoch_lineEdit.setText("当前选中模型无法使用该参数")
            self.set_K_lineEdit.setText("当前选中模型无法使用该参数")
            
            self.learning_rate_lineEdit.setReadOnly(True)
            self.batch_lineEdit.setReadOnly(True)
            self.set_epoch_lineEdit.setReadOnly(True)
            self.set_K_lineEdit.setReadOnly(True)
        elif(selected_text=="贝叶斯"):
            self.learning_rate_lineEdit.setText("当前选中模型无法使用该参数")
            self.batch_lineEdit.setText("当前选中模型无法使用该参数")
            self.set_epoch_lineEdit.setText("当前选中模型无法使用该参数")
            self.set_K_lineEdit.setText("当前选中模型无法使用该参数")
            
            self.learning_rate_lineEdit.setReadOnly(True)
            self.batch_lineEdit.setReadOnly(True)
            self.set_epoch_lineEdit.setReadOnly(True)
            self.set_K_lineEdit.setReadOnly(True)

        elif(selected_text=="FNN" or selected_text=="BiLSTM_CRF"):
            self.select_optimizer.addItems(['SGD','Adam','RMSprop'])
            self.select_kernel.addItems(["relu","sigmoid","tanh"])
            self.learning_rate_lineEdit.clear()
            self.batch_lineEdit.clear()
            self.set_epoch_lineEdit.clear()
            self.set_K_lineEdit.setText("当前选中模型无法使用该参数")

            self.learning_rate_lineEdit.setReadOnly(False)
            self.batch_lineEdit.setReadOnly(False)
            self.set_epoch_lineEdit.setReadOnly(False)
            self.set_K_lineEdit.setReadOnly(True)
        elif(selected_text=="K-means"):
            self.learning_rate_lineEdit.setText("当前选中模型无法使用该参数")
            self.batch_lineEdit.setText("当前选中模型无法使用该参数")
            self.set_epoch_lineEdit.setText("当前选中模型无法使用该参数")
            
            self.learning_rate_lineEdit.setReadOnly(True)
            self.batch_lineEdit.setReadOnly(True)
            self.set_epoch_lineEdit.setReadOnly(True)
            self.set_K_lineEdit.clear()
            self.set_K_lineEdit.setReadOnly(False)
        elif(selected_text=="隐马尔可夫"):
            self.learning_rate_lineEdit.setText("当前选中模型无法使用该参数")
            self.batch_lineEdit.setText("当前选中模型无法使用该参数")
            self.set_epoch_lineEdit.setText("当前选中模型无法使用该参数")
            self.set_K_lineEdit.setText("当前选中模型无法使用该参数")
            
            self.learning_rate_lineEdit.setReadOnly(True)
            self.batch_lineEdit.setReadOnly(True)
            self.set_epoch_lineEdit.setReadOnly(True)
            self.set_K_lineEdit.setReadOnly(True)
            # self.
    def control_model(self):
        self.select_model.clear()
        selected_text = self.select_task.currentText()
        if(selected_text=="文本分类"):
            self.select_model.addItems(["贝叶斯","SVM","FNN"])
        elif(selected_text=="命名实体识别"):
            self.select_model.addItems(["BiLSTM_CRF"])
        elif(selected_text=="文本聚类"):
            self.select_model.addItems(["K-means"])
        elif(selected_text=="序列标注"):
            self.select_model.addItems(["隐马尔可夫"])
    def draw(self):
        train_accuracy=None
        val_accuracy=None
        epoch=self.epoch_data['epoch']
        train_accuracy=self.epoch_data['train_accuracy']
        val_accuracy=self.epoch_data['val_accuracy']
        train_loss=self.epoch_data['train_loss']
        val_loss=self.epoch_data['val_loss']
        self.train_accuracy_series.append(epoch,train_accuracy)
        self.val_accuracy_series.append(epoch,val_accuracy)
        self.train_loss_series.append(epoch,train_loss)
        self.val_loss_series.append(epoch,val_loss)

        if self.flag==1:
            self.axis_y_loss.setRange(0,max(val_loss,train_loss))
            self.flag=2

        # self.evaluation_text_edit.

        self.progress_bar.setValue(epoch*100/int(self.cur_epoch))


    def initUI(self):
        self.setWindowTitle('NLP程序')
        self.setGeometry(400, 200, 1000, 600)
        # 创建网格布局
        self.grid_layout = QGridLayout()
        self.create_first_vbox_layout()
        self.create_second_vbox_layout()
        
        # 将垂直布局添加至网格布局
        self.grid_layout.addLayout(self.vbox_layout_1, 0, 0)
        self.grid_layout.addWidget(QFrame(self, frameShape=QFrame.VLine, frameShadow=QFrame.Sunken),0,1)
        self.grid_layout.addLayout(self.vbox_layout_2, 0, 2)
        self.grid_layout.setColumnStretch(0, 10)  
        self.grid_layout.setColumnStretch(1, 2)  
        self.grid_layout.setColumnStretch(2, 10)  
        # 将布局设置到窗口上
        self.setLayout(self.grid_layout)

        #一些控件信号控制
        self.select_model.currentIndexChanged[str].connect(self.control_widget)
        self.select_task.currentIndexChanged[str].connect(self.control_model)
    
    @pyqtSlot(object)  # 使用特定类型定义槽函数
    def on_finished(self, result):
        # 处理结果
        if(self.cur_task=="文本分类"):
            self.evaluation_text_edit.setText(">>>多分类前馈神经网络性能评估如下...\n"+result[0])
            if(self.cur_model=="FNN"):
                self.evaluation_text_edit.append('\n>>>评分（测试集上损失值和准确率）\n'+""+str(result[4]))
        elif(self.cur_task=="命名实体识别"):
            self.evaluation_text_edit.setText(">>>BiLSTM_CRF模型性能评估如下...\n"+result[0])
        elif(self.cur_task=="文本聚类"):
            self.evaluation_text_edit.setText(">>>K-means模型性能评估如下...\n"+result[0])
        # else:
        #     self.evaluation_text_edit.setText(">>>隐马尔可夫模型性能评估如下...\n"+result[0])

    
    def start_train_thread(self):
        self.worker = Worker(self.start_train)
        self.worker.finished.connect(self.on_finished)
        # 对于这个示例，不需要连接进度更新信号
        self.thread = threading.Thread(target=self.worker.run_task)
        self.thread.start()

    def start_train(self):
        result=None

        self.flag=1
        self.train_accuracy_series.clear()
        self.val_accuracy_series.clear()
        self.train_loss_series.clear()
        self.val_loss_series.clear()
        self.progress_bar.setValue(0)
        self.cur_file=self.select_file_name
        self.cur_task=self.select_task.currentText()
        self.cur_model=self.select_model.currentText()
        self.cur_epoch=self.set_epoch_lineEdit.text()
        self.cur_kernel=self.select_kernel.currentText()
        self.cur_rate=self.learning_rate_lineEdit.text()
        self.cur_batch=self.batch_lineEdit.text()
        self.cur_optimizer=self.select_optimizer.currentText()
        self.cur_K=self.set_K_lineEdit.text()

        if(self.cur_model=='FNN' or self.cur_model=='BiLSTM_CRF'):
            self.axis_x.setRange(0,int(self.cur_epoch))
            self.axis_x_loss.setRange(0,int(self.cur_epoch))
            
        
        # self.axis_y.setRange(0,1)

        if(self.cur_task=="文本分类"):
            if(self.cur_model=="贝叶斯"):
                result=classify_Bayes(self)
                self.model={
                    'model':result[5],
                    'vectorizer': result[6],
                    'label_encoder': result[7],
                    'model_type':"贝叶斯"
                }
            elif(self.cur_model=="SVM"):
                result=classify_SVM(self)
                self.model={
                    'model':result[5],
                    'vectorizer': result[6],
                    'label_encoder': result[7],
                    'model_type':"SVM"
                }
            elif(self.cur_model=="FNN"):
                result=classify_FNN(self)
                self.model={
                    'model':result[1],
                    'vectorizer': result[2],
                    'label_encoder': result[3],
                    'model_type':"FNN"
                }
        elif(self.cur_task=="文本聚类"):
            if(self.cur_model=="K-means"):
                result=cluster_K_means(self)
                self.model={
                    'vectorizer':result[1],
                    'transformer':result[2],
                    'svd':result[3],
                    'normalizer':result[4],
                    'kmeans':result[5]
                }
        elif(self.cur_task=="命名实体识别"):
            if(self.cur_model=="BiLSTM_CRF"):
                result=NER_2(self)
                self.model={
                    'net':result[1],
                    'word_to_id':result[2],
                    'hidden_size':result[3],
                }
                
        else:
            result=train(self)
            self.model=result
        return result

    def save_model(self):
        # 假设 self.model 是你的模型对象
        if not self.model:
            QMessageBox.information(self, "保存失败", "未训练模型", QMessageBox.Ok)
            return 
        file_name=None
        model=self.model
        if(self.cur_task=="命名实体识别"):
            file_name, _ = QFileDialog.getSaveFileName(self, "保存模型", "", "模型文件 (*.tar)")
            if file_name:
                try:
                    torch.save(self.model, file_name)

                    print("模型已保存到:", file_name)
                    QMessageBox.information(self, "成功", "模型已成功保存到：" + file_name, QMessageBox.Ok)
                except Exception as e:
                    QMessageBox.warning(self, "错误", "保存模型失败：" + str(e), QMessageBox.Ok)

        else:
            file_name, _ = QFileDialog.getSaveFileName(self, "保存模型", "", "模型文件 (*.pkl)")
            if file_name:
                try:
                    joblib.dump(self.model, file_name)
                    print("模型已保存到:", file_name)
                    QMessageBox.information(self, "成功", "模型已成功保存到：" + file_name, QMessageBox.Ok)
                except Exception as e:
                    QMessageBox.warning(self, "错误", "保存模型失败：" + str(e), QMessageBox.Ok)

    def select_model_2(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self,"选择文件", "", "所有文件 (*);;模型文件 (*.pkl);", options=options)
        if(self.check_model_format(file_name)):
            self.cur_model_2=file_name
            self.select_model_label_2.setText(os.path.basename(file_name))
    

    def select_model_3(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self,"选择文件", "", "所有文件 (*);;模型文件 (*.tar);", options=options)
        if(self.check_model_format(file_name)):
            self.cur_model_3=file_name
            self.select_model_label_3.setText(os.path.basename(file_name))
    
    
    def select_model_4(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self,"选择文件", "", "所有文件 (*);;模型文件 (*.pkl);", options=options)
        if(self.check_model_format(file_name)):
            self.cur_model_4=file_name
            self.select_model_label_4.setText(os.path.basename(file_name))
    
    def select_model_5(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "所有文件 (*);;模型文件 (*.pkl);", options=options)
        if(self.check_model_format(file_name)):
            self.cur_model_5=file_name
            self.select_model_label_5.setText(os.path.basename(file_name))

    def use_model_2(self):
        param = joblib.load(self.cur_model_2)
        text=self.input_edit.toPlainText()
        article = [text]
        # print(article)
        article_partition, tag_sequence = word_partition(param, article)
        # 打印分词结果及相应的状态标签序列
        HMM_print(article_partition,tag_sequence)

        gold_standard_file='gold_standard.txt'
        predicted_file='predicted.txt'
        # 打开文件，写入文本内容，然后关闭文件
        # 使用结巴进行分词
        words = jieba.cut(text)
        words_space_joined = ' '.join(words)
        with open(gold_standard_file, 'w', encoding='utf-8') as file:
            file.write(words_space_joined)
        # 将标准答案和预测结果转换为BMES标签格式
        convert_to_BMES(gold_standard_file, "BMES_tags.txt")
        convert_to_BMES(predicted_file, "test-xulie.txt")
        # 读取BMES标签格式的文件
        test_seq = read_file("test-xulie.txt")
        standard_seq = read_file("BMES_tags.txt")
        evaluate_HMM(test_seq,standard_seq)

        standard=read_file_to_string(gold_standard_file)
        predict=read_file_to_string(predicted_file)
        # standard_xulie=read_file_to_string('BMES_tags.txt')
        # predict_xulie=read_file_to_string('test-xulie.txt')
        
        self.sequence_labeling.setText("jieba分词结果：\n"+standard+'\n预测分词结果：\n'+predict+'\n')

    def use_model_3(self):
        text=self.input_edit.toPlainText()
        # use_cuda = torch.cuda.is_available() # 检测是否有可用的gpu
        # device = torch.device("cuda:0" if (use_cuda and ngpu>0) else "cpu")
        # checkpoint = str(self.cur_model_3)
        # print(checkpoint)
        # # 加载模型
        # reloaded_model = BiLSTM_CRF(len(word_to_id), hidden_size)
        # reloaded_model = reloaded_model.to(device)
        # if ngpu > 1:
        #     reloaded_model = torch.nn.DataParallel(reloaded_model, device_ids=list(range(ngpu)))  # 设置并行执行

        # print('*' * 27, 'Loading model weights...')
        # # ckpt = torch.load(checkpoint, map_location=device)  # dict  save在CPU 加载到GPU
        # ckpt = torch.load(checkpoint)  # dict  save在GPU 加载到 GPU
        # model_sd = ckpt['net']
        # if device.type == 'cuda' and ngpu > 1:
        #     reloaded_model.module.load_state_dict(model_sd)
        # else:
        #     reloaded_model.load_state_dict(model_sd)
        # print('*' * 27, 'Model loaded success!')

        # reloaded_model.eval()  # 设置eval mode

        # sentences = [
        #         '据中新网报道,由国家一级导演郑方南编剧､执导,夏侯镔､巍子､陶慧敏､王强､侯梦莎､关亚军､王霏､高志强等主演的军旅题材剧《陆军一号》将于1月14日登陆央视综合频道｡该剧聚焦国防和中国陆军改革,展示出新时期中国军人的使命担当和家国情怀｡',
        #         '中国海军网:3月17日下午,南中国海某海域浪花飞卷｡14时30分,千岛湖综合补给舰与正在执行马航失联客机搜救任务的船坞登陆舰井冈山舰､昆仑山舰,导弹护卫舰绵阳舰和永兴岛号远洋救生船汇合',
        #         '美国目前在三个空军基地测试F-35系列战机,分别是佛罗里达州的艾格林空军基地､内华达州的内利斯空军基地､加利福尼亚州的爱德华兹空军基地｡其中,美国于近期组建的首支F-35B飞行中队部署在艾格林空军基地｡',
        #         '东海狂澜骤起,烈日沙场点兵｡近期,第73集团军某旅组织炮兵分队在东海之滨展开对海上目标实弹射击演练,有效检验炮兵分队快速反应､机动投送､高效指挥､精准打击和综合保障能力,全面提升实战化训练水平和遂行任务能力｡'
        # ]

        # for sentence in sentences:
        #     pred_ids, pred_tags = predict(reloaded_model, sentence, word_to_id)
        #     pred_ner = get_entity(pred_tags, pred_ids, sentence)  # 抽取实体
        #     print('sentence:', sentence)
        #     print('predict_ner:', pred_ner, '\n')



    def use_model_4(self):
        # 加载之前训练好的模型
        all_model = joblib.load(self.cur_model_4)
        vectorizer=all_model['vectorizer']
        transformer=all_model['transformer']
        svd=all_model['svd']
        normalizer=all_model["normalizer"]
        kmeans=all_model["kmeans"]
        text=self.input_edit.toPlainText()
        # 待预测的文本
        text_to_predict = [text]
        # 将待预测文本转换为DataFrame，以便于处理
        df_predict = pd.DataFrame({'content': text_to_predict})
        # 预处理文本
        df_predict['segment'] = df_predict['content'].apply(chinese_word_cut)
        # 特征提取
        X_predict = vectorizer.transform(df_predict['segment'])
        # 降维
        X_predict = svd.transform(X_predict)
        # 标准化
        X_predict = normalizer.transform(X_predict)
        # 预测
        labels_predict = kmeans.predict(X_predict)
        # 输出预测结果
        labels_predict[0]=transform_number(labels_predict[0])
        self.cluster.setText(str(labels_predict))

    def use_model_5(self):
        # 加载之前训练好的模型
        all_model = joblib.load(self.cur_model_5)
        model=all_model['model']
        count=all_model['vectorizer']
        le=all_model['label_encoder']
        model_type=all_model["model_type"]
        # 新文本数据
        new_text=self.input_edit.toPlainText()
        # 中文文本分词
        new_text_word = jieba.cut(new_text)
        new_text_cut = ' '.join(word for word in new_text_word)
        new_text_cut = [new_text_cut]
        # 文本数据转换成数值矩阵
        # 使用之前训练数据集使用的 CountVectorizer 实例进行转换
        new_text_count = count.transform(new_text_cut)
        new_text_count = new_text_count.toarray()  # 转换为数组格式
        # 使用训练好的模型进行预测
        if model_type!='FNN':
            predicted_category = model.predict(new_text_count)
            predicted_category_name = le.inverse_transform(predicted_category)
            # 打印预测结果
            self.classification.setText(str(predicted_category_name))
        else:
            predicted_category = np.argmax(model.predict(new_text_count),axis=1)
            predicted_category_name = le.inverse_transform(predicted_category)
            # 打印预测结果
            self.classification.setText(str(predicted_category_name))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
