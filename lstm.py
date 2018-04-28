import argparse
import os
import sys
import pathlib
import datetime
import logging

import tensorflow as tf
import numpy as np


# constants
BOS = '<bos>'
EOS = '<eos>'
PAD = '<pad>'
UNK = '<unk>'


def get_exp_path():
    '''Return new experiment path.'''

    return 'log/exp-{0}'.format(
        datetime.datetime.now().strftime('%m-%d-%H:%M:%S'))


def get_logger(path):
    '''Get logger for experiment.'''

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - '
                                  '%(levelname)s - %(message)s')

    # stdout log
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # file log
    handler = logging.FileHandler(path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # log filter
    logger.addFilter(WarningFilter())

    return logger


def load_corpus(path, param):
    '''Load training corpus.

    Args:
        path: path to corpus
        param: experiment parameters

    Returns:
        A list of sentences.
    '''

    logger = logging.getLogger(__name__)
    logger.info('Loading corpus from %s' % path)
    corpus = []
    line_cnt = 0
    with open(path, 'r') as f:
        for line in f:
            line_cnt += 1
            if (param.max_line_cnt is not None and
                line_cnt >= param.max_line_cnt):
                logger.info('Reach maximum line count %d' %
                                  param.max_line_cnt)
                break

            tokens = [BOS]
            tokens.extend(line.strip().split())
            tokens.append(EOS)

            if len(tokens) > param.max_sentence_length:
                continue

            # extra <PAD> at the end for lstm prediction
            tokens.extend([PAD] * (param.max_sentence_length + 1 - len(tokens)))
            corpus.append(tokens)

    logger.info('Corpus loaded')
    logger.info('%d sentences in original corpus' % line_cnt)
    logger.info('%d sentences returned' % len(corpus))

    return corpus


def build_dictionary(corpus, param):
    '''Build dictionary from corpus.

    Args:
        corpus: corpus on which dictionary is built
        param: experiment parameters

    Returns:
        A dictionary mapping token to index
    '''

    logger = logging.getLogger(__name__)
    logger.info('Building dictionary from training corpus')
    dico, token_cnt = {}, {}
    dico[BOS], dico[EOS], dico[PAD], dico[UNK] = 0, 1, 2, 3
    dico_size = len(dico)

    # count tokens
    for sentence in corpus:
        for token in sentence:
            # skip BOS/EOS/PAD
            if token in dico:
                continue
            cnt = token_cnt.get(token, 0)
            token_cnt[token] = cnt + 1
    
    for token in sorted(token_cnt.keys(),
        key=lambda k: token_cnt[k], reverse=True):
        dico[token] = dico_size
        dico_size += 1
        if dico_size == param.vocab_size:
            break
    
    logger.info('Final size of dictionary is %d' % len(dico))
    return dico


def transform_corpus(corpus, dico, param):
    '''Transform a corpus using a dictionary.
    
    Args:
        corpus: a list of tokenized sentences
        dico: a mapping from token to index
        param: experiment parameters

    Returns:
        A transformed corpus as numpy array
    '''

    logger = logging.getLogger(__name__)
    logger.info('Transforming corpus of size %d' % len(corpus))
    transformed_corpus = []
    for sentence in corpus:
        transformed_sentence = []
        for token in sentence:
            transformed_sentence.append(dico.get(token, dico[UNK]))
        transformed_corpus.append(transformed_sentence)

    logger.info('Finished transforming corpus')
    transformed_corpus = np.array(transformed_corpus, dtype=np.int32)
    return transformed_corpus


def load_pretrained_embeddings(path, param):
    '''Load pretrained word embeddings.

    Args:
        path: path to pretrained word embeddings
        param: experiment parameters

    Returns:
        A `Tensor` with shape [dico_size, emb_dim]
    '''

    logger = logging.getLogger(__name__)
    logger.info('Loading pretrained embedding from %s' % param.pretrained)

    # read embedding
    logger.info('Reading file')
    embedding = np.empty(
        shape=[param.dico_size, param.emb_dim], dtype=np.float)
    found_tokens = set()
    with open(param.pretrained, 'r') as f:
        for i, line in enumerate(f):
            # early break
            if (param.max_pretrained_vocab_size is not None and
                i > param.max_pretrained_vocab_size):
                logger.info('Reach maximum pretrain vocab size %d' %
                            param.max_pretrained_vocab_size)
                break

            line = line.strip().split()
            if i == 0: # first line
                assert len(line) == 2, 'Invalid format at first line'
                _, dim = map(int, line)
                assert dim == param.emb_dim, 'Config to load embedding of ' \
                    'dimension %d but see %d' % (
                    param.emb_dim, dim)
            else: # embedding line
                token = line[0]
                token_id = param.dico.get(token, -1)
                if token_id < 0:
                    continue
                found_tokens.add(token)
                embedding[token_id] = np.array(
                    list(map(float, line[1:])), dtype=np.float)

    # check unfound tokens
    logger.info('Checking unfound tokens')
    for token in param.dico.keys() - found_tokens:
        logger.info('Cannot load pretrained embedding for token %s' % token)
        embedding[param.dico[token]] = np.random.uniform(low=-0.25, high=0.25, size=param.emb_dim)

    logger.info('Finish loading pretrained embedding')
    return tf.convert_to_tensor(embedding, dtype=tf.float32)


def model_fn(features, labels, mode, params):
    '''LSTM model function.

    Args:
        features: model input as token indices,
                  of shape [batch_size, max_sentence_length]
        labels: ground truth output as indices, same shape as `features`
        mode: model mode (training/evaluation/prediction)
        params: dictionary of hyperparameters

    Returns:
        An `EstimatorSpec`
    '''

    logger = logging.getLogger(__name__)
    logger.info('model_fn in mode %s' % str(mode))

    initializer = tf.contrib.layers.xavier_initializer()

    # transposed features and labels
    features = tf.transpose(features['x'])
    batch_size = tf.shape(features)[1]
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.transpose(labels)

    # embedding params
    if params['pretrained'] is not None and mode == tf.estimator.ModeKeys.TRAIN:
        logger.info('Use pretrained embedding at %s' % params['pretrained'])
        embedding_params = tf.Variable(
            initial_value=load_pretrained_embeddings(
                path=params['pretrained'],
                param=params['param']
            ),
            trainable=True
        )
    else:
        if params['pretrained'] is None:
            logger.info('No pretrained embedding')
        else:
            logger.info('In EVAL mode, loading fine tuned embedding')
        embedding_params = tf.Variable(
            initial_value=initializer(
                shape=[params['dico_size'], params['emb_dim']]
            ),
            trainable=True
        )

    # embedding layer
    embeddings = tf.nn.embedding_lookup(
        params=embedding_params,
        ids=features
    )

    # lstm cell
    lstm = tf.contrib.rnn.LSTMCell(
        num_units=params['state_dim'],
        initializer=initializer
    )

    # initial state
    c0 = tf.Variable(initializer(shape=[1, params['state_dim']]))
    h0 = tf.Variable(initializer(shape=[1, params['state_dim']]))
    cell_state = tf.tile(c0, [batch_size, 1])
    hidden_state = tf.tile(h0, [batch_size, 1])
    state = {'state': (cell_state, hidden_state)}

    # unroll lstm
    def lstm_step(current_step):
        output, state['state'] = lstm(current_step, state['state'])
        return output
    output = tf.map_fn(fn=lstm_step, elems=embeddings)

    # output projection
    output = tf.reshape(output, shape=[-1, params['state_dim']])
    if params['hidden_proj_dim'] is not None:
        logger.info('Project lstm output to %d dim' %
                    params['hidden_proj_dim'])
        output = tf.layers.dense(
            inputs=output,
            units=params['hidden_proj_dim'],
            activation=tf.nn.relu,
            kernel_initializer=initializer
        )

    # logits
    logits = tf.layers.dense(
        inputs=output,
        units=params['dico_size'],
        kernel_initializer=initializer
    )
    logits = tf.reshape(
        logits,
        shape=[params['max_sentence_length'], -1, params['dico_size']]
    )

    # prediction
    predict_index = tf.argmax(input=logits, axis=2)
    predict_probability = tf.nn.softmax(logits, name='softmax')
    predictions = {
        'index': predict_index,
        'probability': tf.reduce_max(predict_probability, axis=2),
    }

    # prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # training mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        grads_and_vars = optimizer.compute_gradients(loss)
        tgrads, tvars = zip(*grads_and_vars)
        tgrads, _ = tf.clip_by_global_norm(tgrads, params['max_grad_norm'])
        grads_and_vars = zip(tgrads, tvars)
        train_op = optimizer.apply_gradients(
            grads_and_vars=grads_and_vars,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # perplexity
    label_probability = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(logits, shape=[-1, params['dico_size']]),
        labels=tf.reshape(labels, shape=[-1])
    )
    label_probability = tf.reshape(
        label_probability, shape=tf.shape(labels), name='label_probability')
    pad_id = tf.tile(
        tf.constant(params['dico'][PAD], shape=[1, 1]),
        multiples=[tf.shape(labels)[0], tf.shape(labels)[1]]
    )
    mask = tf.cast(tf.not_equal(labels, pad_id), tf.float32, name='mask')
    sentence_log_probability = (tf.reduce_sum(
        mask * label_probability, axis=0) /
        tf.reduce_sum(mask, axis=0))
    perplexity = tf.exp(sentence_log_probability, name='perplexity')
    average_perplexity = tf.metrics.mean(perplexity)

    # evaluation mode
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions['index']
        ),
        'perplexity': tf.contrib.metrics.streaming_concat(perplexity),
        'average_perplexity': average_perplexity
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops
    )


def main():
    '''Main function.'''

    # command line arguments
    parser = argparse.ArgumentParser(description='LSTM model for NLU project 1')
    # network architecture
    parser.add_argument('--emb_dim', type=int, default=100,
                        help='Embedding dimension, default 100')
    parser.add_argument('--state_dim', type=int, default=512,
                        help='LSTM cell hidden state dimension (for c and h), default 512')
    parser.add_argument('--hidden_proj_dim', type=int, default=None,
                        help='Project hidden output before softmax, default None')
    # input data preprocessing
    parser.add_argument('--train_corpus', type=str, default='data/sentences.train',
                        help='Path to training corpus')
    parser.add_argument('--eval_corpus', type=str, default='data/sentences.eval',
                        help='Path to evaluation corpus')
    parser.add_argument('--max_line_cnt', type=int, default=None,
                        help='Maximum number of lines to load, default None')
    parser.add_argument('--max_sentence_length', type=int, default=30,
                        help='Maximum sentence length in consider, default 30')
    parser.add_argument('--vocab_size', type=int, default=20000,
                        help='Vocabulary size, default 20000')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained word embedding, default None')
    parser.add_argument('--max_pretrained_vocab_size', type=int, default=None,
                        help='Maximum pretrained tokens to read, default None')
    # training
    parser.add_argument('--max_grad_norm', type=float, default=5.0,
                        help='Clip gradient norm to this value, default 5.0')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size, default 64')
    parser.add_argument('--n_epoch', type=int, default=10,
                        help='Training epoch number, default 10')
    parser.add_argument('--exp_path', type=str, default=None,
                        help='Experiment path')
    param = parser.parse_args()

    # parameter validation
    if param.pretrained is not None:
        assert os.path.exists(param.pretrained)
    assert param.vocab_size > 4  # <bos>, <eos>, <pad>, <unk>

    # experiment path
    if param.exp_path is None:
        param.exp_path = get_exp_path()
    pathlib.Path(param.exp_path).mkdir(parents=True, exist_ok=True)

    # logger
    logger = get_logger(param.exp_path + '/experiment.log')
    logger.info('Start of experiment')
    logger.info('============ Initialized logger ============')
    logger.info('\n\t' + '\n\t'.join('%s: %s' % (k, str(v))
                                    for k, v in sorted(dict(vars(param)).items())))

    # load corpus
    train_corpus = load_corpus(param.train_corpus, param)
    eval_corpus = load_corpus(param.eval_corpus, param)

    # build dictionary
    dico = build_dictionary(train_corpus, param)
    param.dico = dico
    param.dico_size = len(dico)

    # transform corpus
    train_corpus = transform_corpus(train_corpus, dico, param)
    eval_corpus = transform_corpus(eval_corpus, dico, param)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_corpus[:,:-1]},
        y=train_corpus[:,1:],
        batch_size=param.batch_size,
        num_epochs=param.n_epoch,
        shuffle=True,
    )
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': eval_corpus[:,:-1]},
        y=eval_corpus[:,1:],
        batch_size=param.batch_size,
        num_epochs=1,
        shuffle=False
    )

    # build training parameters
    model_params = {'param': param}
    for k, v in dict(vars(param)).items():
        model_params[k] = v

    # build model
    logger.info('Building estimator')
    language_model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=param.exp_path,
        params=model_params
    )

    # logging hooks
    tensors_to_log = {'loss': "sparse_softmax_cross_entropy_loss/value:0"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)

    # training
    logger.info('Start training')
    language_model.train(
        input_fn=train_input_fn,
        hooks=[logging_hook]
    )

    # evaluation
    logger.info('Start evaluation')
    result = language_model.evaluate(input_fn=eval_input_fn)
    perplexity = result['perplexity']
    np.savetxt('/'.join([param.exp_path, 'eval.perplexity']), perplexity, fmt='%.10f')


class WarningFilter(logging.Filter):
    '''Remove warning message from tensorflow.'''

    def filter(self, record):
        msg = record.getMessage()
        tf_warning = 'retry (from tensorflow.contrib.learn' in msg
        return not tf_warning


if __name__ == '__main__':
    logging.getLogger('tensorflow').addFilter(WarningFilter())
    logging.getLogger('tensorflow').setLevel(logging.DEBUG)
    main()
