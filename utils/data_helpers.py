# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import warnings
warnings.filterwarnings('ignore')

import os
import time
import heapq
import gensim
import logging
import json
import tensorflow as tf

from collections import OrderedDict
from pylab import *
from texttable import Texttable
from gensim.models import word2vec
from tflearn.data_utils import pad_sequences

import torch
from transformers import BertTokenizer,BertModel

def _option(pattern):
    """
    Get the option according to the pattern.
    (pattern 0: Choose training or restore; pattern 1: Choose best or latest checkpoint.)

    Args:
        pattern: 0 for training step. 1 for testing step.
    Returns:
        The OPTION
    """
    if pattern == 0:
        OPTION = input("[Input] Train or Restore? (T/R): ")
        while not (OPTION.upper() in ['T', 'R']):
            OPTION = input("[Warning] The format of your input is illegal, please re-input: ")
    if pattern == 1:
        OPTION = input("Load Best or Latest Model? (B/L): ")
        while not (OPTION.isalpha() and OPTION.upper() in ['B', 'L']):
            OPTION = input("[Warning] The format of your input is illegal, please re-input: ")
    return OPTION.upper()


def logger_fn(name, input_file, level=logging.INFO):
    """
    The Logger.

    Args:
        name: The name of the logger
        input_file: The logger file path
        level: The logger level
    Returns:
        The logger
    """
    tf_logger = logging.getLogger(name)
    tf_logger.addHandler(logging.StreamHandler()) #自己加的
    tf_logger.setLevel(
        level)
    print(input_file)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(input_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    tf_logger.addHandler(fh)
    return tf_logger


def tab_printer(args, logger):
    """
    Function to print the logs in a nice tabular format.

    Args:
        args: Parameters used for the model.
        logger: The logger
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    t.add_rows([["Parameter", "Value"]])
    logger.info('\n' + t.draw())


def get_out_dir(option, logger):
    """
    Get the out dir.

    Args:
        option: Train or Restore
        logger: The logger
    Returns:
        The output dir
    """
    if option == 'T':
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        logger.info("Writing to {0}\n".format(out_dir))
    if option == 'R':
        MODEL = input("[Input] Please input the checkpoints model you want to restore, "
                      "it should be like (1490175368): ")  # The model you want to restore

        while not (MODEL.isdigit() and len(MODEL) == 10):
            MODEL = input("[Warning] The format of your input is illegal, please re-input: ")
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
        logger.info("Writing to {0}\n".format(out_dir))
    return out_dir


def get_model_name():
    """
    Get the model name used for test.

    Returns:
        The model name
    """
    MODEL = input("[Input] Please input the model file you want to test, it should be like (1490175368): ")

    while not (MODEL.isdigit() and len(MODEL) == 10):
        MODEL = input("[Warning] The format of your input is illegal, "
                      "it should be like (1490175368), please re-input: ")
    return MODEL


def create_prediction_file(output_file, data_id, all_labels, all_predict_labels, all_predict_scores):
    """
    Create the prediction file.

    Args:
        output_file: The all classes predicted results provided by network
        data_id: The data record id info provided by class Data
        all_labels: The all origin labels
        all_predict_labels: The all predict labels by threshold
        all_predict_scores: The all predict scores by threshold
    Raises:
        IOError: If the prediction file is not a .json file
    """
    if not output_file.endswith('.json'):
        raise IOError("[Error] The prediction file is not a json file."
                      "Please make sure the prediction data is a json file.")
    with open(output_file, 'w') as fout:
        data_size = len(all_predict_labels)
        for i in range(data_size):
            predict_labels = [int(i) for i in all_predict_labels[i]]
            predict_scores = [round(i, 4) for i in all_predict_scores[i]]
            labels = [int(i) for i in all_labels[i]]
            data_record = OrderedDict([
                ('id', data_id[i]),
                ('labels', labels),
                ('predict_labels', predict_labels),
                ('predict_scores', predict_scores)
            ])
            fout.write(json.dumps(data_record, ensure_ascii=False) + '\n')


def get_onehot_label_threshold(scores, threshold=0.5):
    """
    Get the predicted onehot labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.

    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_onehot_labels: The predicted labels (onehot)
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        onehot_labels_list = [0] * len(score)
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                onehot_labels_list[index] = 1
                count += 1
        if count == 0:
            max_score_index = score.index(max(score))
            onehot_labels_list[max_score_index] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def get_onehot_label_topk(scores, top_num=1):
    """
    Get the predicted onehot labels based on the topK number.

    Args:
        scores: The all classes predicted scores provided by network
        top_num: The max topK number (default: 5)
    Returns:
        predicted_onehot_labels: The predicted labels (onehot)
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        onehot_labels_list = [0] * len(score)
        max_num_index_list = list(map(score.index, heapq.nlargest(top_num, score)))
        for i in max_num_index_list:
            onehot_labels_list[i] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def get_label_threshold(scores, threshold=0.5):
    """
    Get the predicted labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.

    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_labels: The predicted labels
        predicted_scores: The predicted scores
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        index_list = []
        score_list = []
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                index_list.append(index)
                score_list.append(predict_score)
                count += 1
        if count == 0:
            index_list.append(score.index(max(score)))
            score_list.append(max(score))
        predicted_labels.append(index_list)
        predicted_scores.append(score_list)
    return predicted_labels, predicted_scores


def get_label_topk(scores, top_num=1):
    """
    Get the predicted labels based on the topK number.

    Args:
        scores: The all classes predicted scores provided by network
        top_num: The max topK number (default: 5)
    Returns:
        The predicted labels
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        score_list = []
        index_list = np.argsort(score)[-top_num:]
        index_list = index_list[::-1]
        for index in index_list:
            score_list.append(score[index])
        predicted_labels.append(np.ndarray.tolist(index_list))
        predicted_scores.append(score_list)
    return predicted_labels, predicted_scores


def create_metadata_file(word2vec_file, output_file):
    """
    Create the metadata file based on the corpus file (Used for the Embedding Visualization later).

    Args:
        word2vec_file: The word2vec file
        output_file: The metadata file path
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist.")

    BERT_path_root = "./data/chinese-roberta-wwm-ext-large"
    tokenizer = BertTokenizer.from_pretrained(BERT_path_root)
    model = BertModel.from_pretrained(BERT_path_root)
    EXAMPLE_SENTENCE = "你好，我的名字是吳曉光"
    encodes = tokenizer.encode(EXAMPLE_SENTENCE, add_special_tokens=True)
    print(encodes)

    # model = gensim.models.Word2Vec.load(word2vec_file)
    word2idx = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    word2idx_sorted = [(k, word2idx[k]) for k in sorted(word2idx, key=word2idx.get, reverse=False)]

    with open(output_file, 'w+') as fout:
        for word in word2idx_sorted:
            if word[0] is None:
                print("[Warning] Empty Line, should replaced by any thing else, or will cause a bug of tensorboard")
                fout.write('<Empty Line>' + '\n')
            else:
                fout.write(word[0] + '\n')


def load_word2vec_matrix(word2vec_file):
    """
    Return the word2vec model matrix.

    Args:
        word2vec_file: The word2vec file
    Returns:
        The word2vec model matrix
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist. ")

    # model = gensim.models.Word2Vec.load(word2vec_file)
    # vocab_size = model.wv.vectors.shape[0]
    # embedding_size = model.vector_size  # embedding_size = 256
    # vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    # embedding_matrix = np.zeros([vocab_size, embedding_size])
    # for key, value in vocab.items():
    #     if key is not None:
    #         embedding_matrix[value] = model[key]
    # return vocab_size, embedding_size, embedding_matrix

    # BERT
    BERT_path_root = "./data/chinese-roberta-wwm-ext-large"
    tokenizer = BertTokenizer.from_pretrained(BERT_path_root)
    bert = BertModel.from_pretrained(BERT_path_root)
    token_embedding = {token: bert.get_input_embeddings()(torch.tensor(id))  for token, id in tokenizer.get_vocab().items()}
    vocab_size = len(token_embedding)

    embedding_size = len(token_embedding['[CLS]'])
    embedding_matrix = np.zeros([vocab_size, embedding_size])
    for token, id in tokenizer.get_vocab().items():
        embedding_matrix[id] = token_embedding[token].detach().numpy()

    return vocab_size, embedding_size, embedding_matrix


def bert_encode(data,maximum_length) :
    input_ids = []
    attention_masks = []

    for i in range(len(data)):
        encoded = tokenizer.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=maximum_length,
            pad_to_max_length=True,
            return_attention_mask=True,
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids),np.array(attention_masks)


def data_word2vec(input_file, num_classes_list, total_classes, word2vec_model):
    """
    Create the research data tokenindex based on the word2vec model file.
    Return the class Data(includes the data tokenindex and data labels).

    Args:
        input_file: The research data
        num_classes_list: <list> The number of classes
        total_classes: The total number of classes
        word2vec_model: The word2vec model file
    Returns:
        The Class _Data() (includes the data tokenindex and data labels)
    Raises:
        IOError: If the input file is not the .json file
    """
    # w2v
    # vocab = dict([(k, v.index) for (k, v) in word2vec_model.wv.vocab.items()])

    # #　# BERT
    BERT_path_root = "./data/chinese-roberta-wwm-ext-large"
    tokenizer = BertTokenizer.from_pretrained(BERT_path_root)
    bert = BertModel.from_pretrained(BERT_path_root)
    # token_embedding = {token: bert.get_input_embeddings()(torch.tensor(id))  for token, id in tokenizer.get_vocab().items()}
    vocab = dict([(token, id) for token, id in tokenizer.get_vocab().items()])

    def _token_to_index(content): ## change the vocab part 
        result = []
        for item in content:
            word2id = vocab.get(item)
            if word2id is None:
                word2id = 0
            result.append(word2id)
        return result

    # def _token_to_index(content): ## BERT sentence option
    #     input_ids = []
    #     attention_masks = []
    #     for i in range(len(data)):
    #         encoded = tokenizer.encode_plus(
    #             data[i],
    #             add_special_tokens=True,
    #             max_length=128,
    #             pad_to_max_length=True,
    #             return_attention_mask=True,
    #         )
        
    #         input_ids.append(encoded['input_ids'])
    #         attention_masks.append(encoded['attention_mask'])

    #     return np.array(input_ids)


    def _create_onehot_labels(labels_index, num_labels): ## no effect 
        label = [0] * num_labels
        for item in labels_index:
            label[int(item)] = 1
        return label

    if not input_file.endswith('.json'):
        raise IOError("[Error] The research data is not a json file. "
                      "Please preprocess the research data into the json file.")
    with open(input_file,encoding='utf-8') as fin:
        id_list = []
        title_index_list = []
        abstract_index_list = []
        abstract_content_list = []
        labels_list = []
        onehot_labels_list = []
        onehot_labels_tuple_list = []
        total_line = 0

        bert_content_list = []

        for eachline in fin:
            data = json.loads(eachline)
            patent_id = data['id']
            title_content = data['title']
            abstract_content = data['abstract']
            ## expeirment
            # bert_sentence =[] ## single words option   =>  bugged 
            bert_sentence ="" ## sentence option
            for words in abstract_content:
                bert_sentence+=words
            bert_content_list.append(bert_sentence)

            first_labels = data['section']
            second_labels = data['subsection']
            third_labels = data['group']
            #fourth_labels = data['subgroup']
            total_labels = data['labels']

            id_list.append(patent_id)
            title_index_list.append(_token_to_index(title_content))

            # abstract_index_list.append(_token_to_index(abstract_content))
            # abstract_content_list.append(abstract_content)
            abstract_index_list.append(_token_to_index(bert_content_list))
            abstract_content_list.append(bert_content_list)

            labels_list.append(total_labels)
            # labels_tuple = (_create_onehot_labels(first_labels, num_classes_list[0]),
            #                 _create_onehot_labels(second_labels, num_classes_list[1]),
            #                 _create_onehot_labels(third_labels, num_classes_list[2]))
            #                 _create_onehot_labels(fourth_labels, num_classes_list[3]))
            labels_tuple = (_create_onehot_labels(first_labels, num_classes_list[0]),
                            _create_onehot_labels(second_labels, num_classes_list[1]),
                            _create_onehot_labels(third_labels, num_classes_list[2]))

        
           
            onehot_labels_tuple_list.append(labels_tuple)
            onehot_labels_list.append(_create_onehot_labels(total_labels, total_classes))
            total_line += 1

    class _Data:
        def __init__(self):
            pass

        @property
        def number(self):
            return total_line

        @property
        def patent_id(self):
            return id_list

        @property
        def title_tokenindex(self):
            return title_index_list

        @property
        def abstract_tokenindex(self):
            return abstract_index_list

        @property
        def abstract_content(self):
            return abstract_content_list

        @property
        def labels(self):
            return labels_list

        @property
        def onehot_labels_tuple(self):
            return onehot_labels_tuple_list

        @property
        def onehot_labels(self):
            return onehot_labels_list

    return _Data()


def data_augmented(data, drop_rate=1.0):
    """
    Data augment.

    Args:
        data: The Class _Data()
        drop_rate: The drop rate
    Returns:
        The Class _AugData()
    """
    aug_num = data.number
    aug_patent_id = data.patent_id
    aug_title_tokenindex = data.title_tokenindex
    aug_abstract_tokenindex = data.abstract_tokenindex
    aug_labels = data.labels
    aug_onehot_labels = data.onehot_labels
    aug_onehot_labels_tuple = data.onehot_labels_tuple

    for i in range(len(data.aug_abstract_tokenindex)):
        data_record = data.tokenindex[i]
        if len(data_record) == 1:  # 句子长度为 1，则不进行增广
            continue
        elif len(data_record) == 2:  # 句子长度为 2，则交换两个词的顺序
            data_record[0], data_record[1] = data_record[1], data_record[0]
            aug_patent_id.append(data.patent_id[i])
            aug_title_tokenindex.append(data.title_tokenindex[i])
            aug_abstract_tokenindex.append(data_record)
            aug_labels.append(data.labels[i])
            aug_onehot_labels.append(data.onehot_labels[i])
            aug_onehot_labels_tuple.append(data.onehot_labels_tuple[i])
            aug_num += 1
        else:
            data_record = np.array(data_record)
            for num in range(len(data_record) // 10):  # 打乱词的次数，次数即生成样本的个数；次数根据句子长度而定
                # random shuffle & random drop
                data_shuffled = np.random.permutation(np.arange(int(len(data_record) * drop_rate)))
                new_data_record = data_record[data_shuffled]

                aug_patent_id.append(data.patent_id[i])
                aug_title_tokenindex.append(data.title_tokenindex[i])
                aug_abstract_tokenindex.append(list(new_data_record))
                aug_labels.append(data.labels[i])
                aug_onehot_labels.append(data.onehot_labels[i])
                aug_onehot_labels_tuple.append(data.onehot_labels_tuple[i])
                aug_num += 1

    class _AugData:
        def __init__(self):
            pass

        @property
        def number(self):
            return aug_num

        @property
        def patent_id(self):
            return aug_patent_id

        @property
        def title_tokenindex(self):
            return aug_title_tokenindex

        @property
        def abstract_tokenindex(self):
            return aug_abstract_tokenindex

        @property
        def labels(self):
            return aug_labels

        @property
        def onehot_labels(self):
            return aug_onehot_labels

        @property
        def onehot_labels_tuple(self):
            return aug_onehot_labels_tuple

    return _AugData()


def load_data_and_labels(data_file, num_classes_list, total_classes, word2vec_file, data_aug_flag):
    """
    Load research data from files, splits the data into words and generates labels.
    Return split sentences, labels and the max sentence length of the research data.

    Args:
        data_file: The research data
        num_classes_list: <list> The number of classes
        total_classes: The total number of classes
        word2vec_file: The word2vec file
        data_aug_flag: The flag of data augmented
    Returns:
        The class _Data()
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    # Load word2vec file
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist. ")
    
    BERT_path_root = "./data/chinese-roberta-wwm-ext-large"
    tokenizer = BertTokenizer.from_pretrained(BERT_path_root)
    model = BertModel.from_pretrained(BERT_path_root)
    # encodes = tokenizer.encode(EXAMPLE_SENTENCE, add_special_tokens=True)

    # #model = Word2Vec.load(word2vec_file)
    # model = word2vec.Word2Vec.load(word2vec_file)

    # Load data from files and split by words
    data = data_word2vec(data_file, num_classes_list, total_classes, model)
    if data_aug_flag:
        data = data_augmented(data)

    # plot_seq_len(data_file, data)

    return data


def pad_data(data, pad_seq_len):
    """
    Padding each sentence of research data according to the max sentence length.
    Return the padded data and data labels.

    Args:
        data: The research data
        pad_seq_len: The max sentence length of research data
    Returns:
        pad_seq: The padded data
        labels: The data labels
    """
    abstract_pad_seq = pad_sequences(data.abstract_tokenindex, maxlen=pad_seq_len, value=0.)
    onehot_labels_list = data.onehot_labels
    onehot_labels_list_tuple = data.onehot_labels_tuple
    return abstract_pad_seq, onehot_labels_list, onehot_labels_list_tuple


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    含有 yield 说明不是一个普通函数，是一个 Generator.
    函数效果：对 data，一共分成 num_epochs 个阶段（epoch），在每个 epoch 内，如果 shuffle=True，就将 data 重新洗牌，
    批量生成 (yield) 一批一批的重洗过的 data，每批大小是 batch_size，一共生成 int(len(data)/batch_size)+1 批。

    Args:
        data: The data
        batch_size: The size of the data batch
        num_epochs: The number of epochs
        shuffle: Shuffle or not (default: True)
    Returns:
        A batch iterator for data set
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
