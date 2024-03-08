# coding=utf-8

# 输入一段话，识别对应的实体

from config import Config
from utils import load_vocab, load_model, InputFeatures
from model.bert_lstm_crf import BERT_LSTM_CRF
import torch
import warnings

warnings.filterwarnings('ignore', category=UserWarning)



config = Config()
label_dic = {"O":0,"B-renwu":1,"I-renwu":2,"<pad>":3}  # {tag: index}
vocab = load_vocab(config.vocab)
def load_ner_model(config, tagset_size):
    model = BERT_LSTM_CRF(config.ernie_path, tagset_size, config.ernie_embedding, config.rnn_hidden, config.rnn_layer,
                           dropout_ratio=config.dropout_ratio, dropout1=config.dropout1, use_cuda=config.use_cuda)
    model = load_model(model)
    if config.use_cuda:
        model.cuda()
    return model

model = load_ner_model(config, len(label_dic))

def encoder_corpus(sentences, max_length, vocab):
    if isinstance(sentences, str):
        sentences = [sentences]
    result = []
    for line in sentences:
        text = line.strip()
        tokens = list(text)
        if len(tokens) > max_length - 2:
            tokens = tokens[0:(max_length - 2)]
        tokens_f = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)
        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=None)
        result.append(feature)
    return result


def predict(sentences):
    print(sentences)
    sentences = sentences.replace(' ', '')
    data = encoder_corpus(sentences, config.max_length, vocab)
    input_ids = torch.LongTensor([temp.input_id for temp in data]) # [[token1_index, token_2_index], []...]
    input_masks = torch.LongTensor([temp.input_mask for temp in data])
    model.eval()
    with torch.no_grad():
        feats = model(input_ids.cuda(), input_masks.cuda())
        best_path = model.crf.decode(feats, input_masks.byte().cuda())
        id2tag = {label_dic[tag]: tag for tag in label_dic.keys()}
        labels = [id2tag[index] for index in best_path[0]]
        sentences = ['[CLS]'] + list(sentences) + ['[SEP]']
        assert len(sentences) == len(labels)
        ss=""
        entities=[]
        for s,lab in zip(sentences,labels):
            if lab[2:]=="entity":
                ss=ss+s
            else:
                if ss!="":
                    entities.append(ss)
                ss=""

        return entities

if __name__ == '__main__':
    sentences = '输入句子'
    entities = predict(sentences)
    print(entities, sep='\n')
    