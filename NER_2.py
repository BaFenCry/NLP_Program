import torch
import torchkeras
from torchcrf import CRF
from tqdm import tqdm
import datetime
import time
import copy
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
from torch.optim import *
from sklearn.metrics import classification_report

ngpu = 1
device = 'cpu'
tag_to_id = tag_to_id = {'<pad>': 0, 'B-PER': 15, 'I-EQU': 1, 'B-FAC': 2, 'I-GPE': 3, 'B-GPE': 4, 'I-LOC': 5, 'B-EQU': 6, 'B-ORG': 7, 'B-LOC': 8, 'I-PER': 9, 'B-TIME': 10, 'I-ORG': 11, 'O': 12, 'I-TIME': 13, 'I-FAC': 14}
id_to_tag = {id: tag for tag, id in tag_to_id.items()}
word_to_id = {'<pad>': 0, '<unk>': 1}
tags_num = len(tag_to_id)
LR = 1e-3
EPOCHS = 20
maxlen = 60
embedding_dim = 100
hidden_size = 128
batch_size = 512
pad_token = '<pad>'
pad_id = 0
unk_token = '<unk>'
unk_id = 1
def predict(model, sentence, word_to_id):
    inp_ids = [word_to_id[w] if w in word_to_id else unk_id for w in sentence]
    inp_ids = torch.tensor(inp_ids, dtype=torch.long).unsqueeze(dim=0)
    # print(inp_ids.shape)  # [56, 60]
    # forward
    logits = model(inp_ids)
    preds = model.module.crf_decode(logits, inp_logits=True)  # List[List]
    pred_ids = preds[0]
    pred_tags = [id_to_tag[tag_id] for tag_id in pred_ids]

    return pred_ids, pred_tags

def get_entity(pred_tags, pred_ids, sentence):
    ner = {'per':[], 'loc':[], 'org':[]}
    i = 0
    while i<len(pred_tags):
        if pred_tags[i]=='O' or pred_ids[i]==0:
            i += 1
        elif pred_tags[i]=='B-PER':
            j = i
            while j+1<len(pred_tags) and pred_tags[j+1]=='I-PER':
                j += 1
            per = [w for w in sentence[i:j+1]]
            ner['per'].append(''.join(per))
            i = j+1
        elif pred_tags[i]=='B-LOC':
            j = i
            while j+1<len(pred_tags) and pred_tags[j+1]=='I-LOC':
                j += 1
            loc = [w for w in sentence[i:j+1]]
            ner['loc'].append(''.join(loc))
            i = j+1
        elif pred_tags[i]=='B-ORG':
            j = i
            while j+1<len(pred_tags) and pred_tags[j+1]=='I-ORG':
                j += 1
            org = [w for w in sentence[i:j+1]]
            ner['org'].append(''.join(org))
            i = j+1
        else:
            i += 1
    return ner
class BiLSTM_CRF(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_id)
        self.bi_lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size // 2, batch_first=True,
                                    bidirectional=True)  # , dropout=0.2)
        self.hidden2tag = torch.nn.Linear(hidden_size, tags_num)
        self.crf = CRF(num_tags=tags_num, batch_first=True)
    def init_hidden(self, batch_size):
        # device = 'cpu'
        _batch_size = batch_size//ngpu
        return (torch.randn(2, _batch_size, self.hidden_size // 2, device=device),
                torch.randn(2, _batch_size, self.hidden_size // 2, device=device))  # ([b=1,2,hidden_size//2], [b=1,2,hidden_size//2])
    def forward(self, inp):  # inp [b, seq_len=60]
        self.bi_lstm.flatten_parameters()
        embeds = self.embedding(inp)
        lstm_out, _ = self.bi_lstm(embeds, None)  
        logits = self.hidden2tag(lstm_out)  
        return logits
    def crf_neg_log_likelihood(self, inp, tags, mask=None, inp_logits=False):  # [b, seq_len, tags_num], [b, seq_len]
        if inp_logits:
            logits = inp
        else:
            logits = self.forward(inp)
        if mask is None:
            mask = torch.logical_not(torch.eq(tags, torch.tensor(0)))  # =>[b, seq_len],每个元素为bool值，如果序列中有pad，则mask相应位置就为False
            mask = mask.type(torch.uint8)
        crf_llh = self.crf(logits, tags, mask, reduction='mean') # Compute the conditional log likelihood of a sequence of tags given emission scores
        return -crf_llh

    def crf_decode(self, inp, mask=None, inp_logits=False):
        if inp_logits:
            logits = inp
        else:
            logits = self.forward(inp)
        if mask is None and inp_logits is False:
            mask = torch.logical_not(torch.eq(inp, torch.tensor(0)))  # =>[b, seq_len],每个元素为bool值，如果序列中有pad，则mask相应位置就为False
            mask = mask.type(torch.uint8)
        return self.crf.decode(emissions=logits, mask=mask)


def get_metrics(y_true, y_pred):
    average = 'weighted'
    print('*'*27, average+'_precision_score:{:.3f}'.format(precision_score(y_true, y_pred, average=average)))
    print('*'*27, average+'_recall_score:{:.3}'.format(recall_score(y_true, y_pred, average=average)))
    print('*'*27, average+'_f1_score:{:.3f}'.format(f1_score(y_true, y_pred, average=average)))

    print('*'*27, 'accuracy:{:.3f}'.format(accuracy_score(y_true, y_pred)))
    print('*'*27, 'confusion_matrix:\n', confusion_matrix(y_true, y_pred))
    print('*'*27, 'classification_report:\n', classification_report(y_true, y_pred))

def NER_2(self):
    print("开始训练")
    ngpu = 1
    device = 'cpu'
    tag_to_id = tag_to_id = {'<pad>': 0, 'B-PER': 15, 'I-EQU': 1, 'B-FAC': 2, 'I-GPE': 3, 'B-GPE': 4, 'I-LOC': 5, 'B-EQU': 6, 'B-ORG': 7, 'B-LOC': 8, 'I-PER': 9, 'B-TIME': 10, 'I-ORG': 11, 'O': 12, 'I-TIME': 13, 'I-FAC': 14}
    id_to_tag = {id: tag for tag, id in tag_to_id.items()}
    word_to_id = {'<pad>': 0, '<unk>': 1}
    tags_num = len(tag_to_id)
    LR = 1e-3
    EPOCHS = 20
    maxlen = 60
    embedding_dim = 100
    hidden_size = 128
    batch_size = 512
    pad_token = '<pad>'
    pad_id = 0
    unk_token = '<unk>'
    unk_id = 1
    def read_data(filepath):
        sentences = []
        tags = []
        with open(filepath, 'r', encoding='utf-8') as f:
            tmp_sentence = []
            tmp_tags = []
            for line in f:
                if line == '\n' and len(tmp_sentence) != 0:
                    assert len(tmp_sentence) == len(tmp_tags)
                    sentences.append(tmp_sentence)
                    tags.append(tmp_tags)
                    tmp_sentence = []
                    tmp_tags = []
                else:
                    line = line.strip().split(' ')
                    tmp_sentence.append(line[0])
                    tmp_tags.append(line[1])
            if len(tmp_sentence) != 0:
                assert len(tmp_sentence) == len(tmp_tags)
                sentences.append(tmp_sentence)
                tags.append(tmp_tags)
        return sentences, tags


    sentences, tags = read_data(self.select_file_name[1])
    def build_vocab(sentences):
        for sentence in sentences:  # 建立word到索引的映射
            for word in sentence:
                if word not in word_to_id:
                    word_to_id[word] = len(word_to_id)
        return word_to_id

    word_to_id = build_vocab(sentences)
    print('vocab size:', len(word_to_id))


    def convert_to_ids_and_padding(seqs, to_ids):
        ids = []
        for seq in seqs:
            if len(seq)>=maxlen: # 截断
                ids.append([to_ids[w] if w in to_ids else unk_id for w in seq[:maxlen]])
            else: # padding
                ids.append([to_ids[w] if w in to_ids else unk_id for w in seq] + [0]*(maxlen-len(seq)))
        return torch.tensor(ids, dtype=torch.long)


    def load_data(filepath, word_to_id, shuffle=False):
        sentences, tags = read_data(filepath)
        inps = convert_to_ids_and_padding(sentences, word_to_id)
        trgs = convert_to_ids_and_padding(tags, tag_to_id)
        inp_dset = torch.utils.data.TensorDataset(inps, trgs)
        inp_dloader = torch.utils.data.DataLoader(inp_dset,
                                                batch_size=int(self.cur_batch),
                                                shuffle=shuffle,
                                                num_workers=4)
        return inp_dloader



    ngpu = 1
    device = 'cpu'

    class BiLSTM_CRF(torch.nn.Module):
        def __init__(self, vocab_size, hidden_size):
            super(BiLSTM_CRF, self).__init__()
            self.hidden_size = hidden_size
            self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_id)
            self.bi_lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size // 2, batch_first=True,
                                        bidirectional=True)  # , dropout=0.2)
            self.hidden2tag = torch.nn.Linear(hidden_size, tags_num)
            self.crf = CRF(num_tags=tags_num, batch_first=True)
        def init_hidden(self, batch_size):
            # device = 'cpu'
            _batch_size = batch_size//ngpu
            return (torch.randn(2, _batch_size, self.hidden_size // 2, device=device),
                    torch.randn(2, _batch_size, self.hidden_size // 2, device=device))  # ([b=1,2,hidden_size//2], [b=1,2,hidden_size//2])
        def forward(self, inp):  # inp [b, seq_len=60]
            self.bi_lstm.flatten_parameters()
            embeds = self.embedding(inp)
            lstm_out, _ = self.bi_lstm(embeds, None)  
            logits = self.hidden2tag(lstm_out)  
            return logits
        def crf_neg_log_likelihood(self, inp, tags, mask=None, inp_logits=False):  # [b, seq_len, tags_num], [b, seq_len]
            if inp_logits:
                logits = inp
            else:
                logits = self.forward(inp)
            if mask is None:
                mask = torch.logical_not(torch.eq(tags, torch.tensor(0)))  # =>[b, seq_len],每个元素为bool值，如果序列中有pad，则mask相应位置就为False
                mask = mask.type(torch.uint8)
            crf_llh = self.crf(logits, tags, mask, reduction='mean') # Compute the conditional log likelihood of a sequence of tags given emission scores
            return -crf_llh

        def crf_decode(self, inp, mask=None, inp_logits=False):
            if inp_logits:
                logits = inp
            else:
                logits = self.forward(inp)
            if mask is None and inp_logits is False:
                mask = torch.logical_not(torch.eq(inp, torch.tensor(0)))  # =>[b, seq_len],每个元素为bool值，如果序列中有pad，则mask相应位置就为False
                mask = mask.type(torch.uint8)
            return self.crf.decode(emissions=logits, mask=mask)

    # 查看模型
    model = BiLSTM_CRF(len(word_to_id), hidden_size)
    torchkeras.summary(model, input_shape=(60,), input_dtype=torch.int64)


    ngpu = 4 # 4张GPU卡
    use_cuda = torch.cuda.is_available() # 检测是否有可用的gpu
    device = torch.device("cuda:0" if (use_cuda and ngpu>0) else "cpu")
    print('*'*8, 'device:', device)
    # 设置评价指标
    metric_func = lambda y_pred, y_true: accuracy_score(y_true, y_pred)
    metric_name = 'acc'
    df_history = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_"+metric_name])


    def train_step(model, inps, tags, optimizer):
        inps = inps.to(device)
        tags = tags.to(device)
        mask = torch.logical_not(torch.eq(inps, torch.tensor(0)))  # =>[b, seq_len]
        model.train()  # 设置train mode
        optimizer.zero_grad()  # 梯度清零
        # forward
        logits = model(inps)
        loss = model.module.crf_neg_log_likelihood(logits, tags, mask=mask, inp_logits=True)
        # backward
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数
        preds = model.module.crf_decode(logits, mask=mask, inp_logits=True) # List[List]
        pred_without_pad = []
        for pred in preds:
            pred_without_pad.extend(pred)
        tags_without_pad = torch.masked_select(tags, mask).cpu().numpy() # 返回是1维张量
        # print('tags_without_pad:', tags_without_pad.shape, type(tags_without_pad)) # [5082] tensor
        metric = metric_func(pred_without_pad, tags_without_pad)
        # print('*'*8, metric) # 标量
        return loss.item(), metric


    @torch.no_grad()
    def validate_step(model, inps, tags):
        inps = inps.to(device)
        tags = tags.to(device)
        mask = torch.logical_not(torch.eq(inps, torch.tensor(0)))  # =>[b, seq_len],每个元素为bool值，如果序列中有pad，则mask相应位置就为False
        model.eval()  # 设置eval mode
        # forward
        logits = model(inps)
        loss = model.module.crf_neg_log_likelihood(logits, tags, mask=mask, inp_logits=True)
        preds = model.module.crf_decode(logits, mask=mask, inp_logits=True)  # List[List]
        pred_without_pad = []
        for pred in preds:
            pred_without_pad.extend(pred)
        tags_without_pad = torch.masked_select(tags, mask).cpu().numpy()  # 返回是1维张量
        metric = metric_func(pred_without_pad, tags_without_pad)
        # print('*' * 8, metric) # 标量
        return loss.item(), metric



    # 打印时间
    def printbar():
        nowtime = datetime.datetime.now().strftime('%Y-%m_%d %H:%M:%S')
        print('\n' + "=========="*8 + '%s'%nowtime)

    temp=None

    def train_model(model, train_dloader, val_dloader, optimizer, num_epochs=10, print_every=150):
        starttime = time.time()
        print('*' * 27, 'start training...')
        printbar()
        best_metric = 0.
        for epoch in range(1, num_epochs+1):
            # 训练
            loss_sum, metric_sum = 0., 0.
            for step, (inps, tags) in enumerate(train_dloader, start=1):
                loss, metric = train_step(model, inps, tags, optimizer)
                loss_sum += loss
                metric_sum += metric
                if step % print_every == 0:
                    print('*'*27, f'[step = {step}] loss: {loss_sum/step:.3f}, {metric_name}: {metric_sum/step:.3f}')

            val_loss_sum, val_metric_sum = 0., 0.
            for val_step, (inps, tags) in enumerate(val_dloader, start=1):
                val_loss, val_metric = validate_step(model, inps, tags)
                val_loss_sum += val_loss
                val_metric_sum += val_metric
            record = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
            df_history.loc[epoch - 1] = record

            # 打印epoch级别日志
            print('EPOCH = {} loss: {:.3f}, {}: {:.3f}, val_loss: {:.3f}, val_{}: {:.3f}'.format(
                record[0], record[1], metric_name, record[2], record[3],metric_name,  record[4]))
            
            self.epoch_data = {
                'epoch': record[0],
                'train_loss': record[1],
                'val_loss': record[3],
                'train_accuracy': record[2],
                'val_accuracy': record[4]
            }
            self.variable_changed.emit()

            printbar()
            current_metric_avg = val_metric_sum/val_step
            if current_metric_avg > best_metric:
                best_metric = current_metric_avg
                if device.type == 'cuda' and ngpu > 1:
                    model_sd = copy.deepcopy(model.module.state_dict())
                else:
                    model_sd = copy.deepcopy(model.state_dict())
                # 保存
                
                # torch.save({
                #     'loss': loss_sum / step,
                #     'epoch': epoch,
                #     'net': model_sd,
                #     'opt': optimizer.state_dict(),
                # }, checkpoint)
                temp=model_sd
            if epoch==1:
                temp=model_sd
                

        endtime = time.time()
        time_elapsed = endtime - starttime
        print('*' * 27, 'training finished...')
        print('*' * 27, 'and it costs {} h {} min {:.2f} s'.format(int(time_elapsed // 3600),
                                                                int((time_elapsed % 3600) // 60),
                                                                (time_elapsed % 3600) % 60))

        print('Best val Acc: {:4f}'.format(best_metric))
        return df_history



    train_dloader = load_data(self.select_file_name[1], word_to_id)
    val_dloader = load_data(self.select_file_name[2], word_to_id)

    model = BiLSTM_CRF(len(word_to_id), hidden_size)
    model = model.to(device)
    if ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(ngpu)))  # 设置并行执行  device_ids=[0,1,2,3]

    optimizer = None
    if self.cur_optimizer=="SGD":
        optimizer=torch.optim.SGD(model.parameters(),lr=float(self.cur_rate))
    elif self.cur_optimizer=="Adam":
        optimizer=torch.optim.Adam(model.parameters(),lr=float(self.cur_rate))
    elif self.cur_optimizer=="RMSprop":
        optimizer=torch.optim.RMSprop(model.parameters(),lr=float(self.cur_rate))
    train_model(model, train_dloader, val_dloader, optimizer, num_epochs=int(self.cur_epoch), print_every=50)

    @torch.no_grad()
    def eval_step(model, inps, tags):
        inps = inps.to(device)
        tags = tags.to(device)
        mask = torch.logical_not(torch.eq(inps, torch.tensor(0)))  # =>[b, seq_len],每个元素为bool值，如果序列中有pad，则mask相应位置就为False
        # forward
        logits = model(inps)
        preds = model.module.crf_decode(logits, mask=mask, inp_logits=True)  # List[List]
        pred_without_pad = []
        for pred in preds:
            pred_without_pad.extend(pred)
        tags_without_pad = torch.masked_select(tags, mask).cpu()  # 返回是1维张量

        return torch.tensor(pred_without_pad), tags_without_pad

    def evaluate(model, test_dloader):
        model.eval()  # 设置eval mode
        starttime = time.time()
        print('*' * 27, 'start evaluating...')
        printbar()
        preds, labels = [], []
        for step, (inps, tags) in enumerate(tqdm(test_dloader), start=1):
            pred, tags = eval_step(model, inps, tags)
            preds.append(pred)
            labels.append(tags)

        y_true = torch.cat(labels, dim=0)
        y_pred = torch.cat(preds, dim=0)
        endtime = time.time()
        print('evaluating costs: {:.2f}s'.format(endtime - starttime))
        
        return y_true.cpu(), y_pred.cpu()

    # 加载测试数据
    test_dloader = load_data(self.select_file_name[0], word_to_id)
    y_true, y_pred = evaluate(model, test_dloader)
    report=classification_report(y_true, y_pred)
    return [report,temp,word_to_id,hidden_size]






