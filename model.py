import tensorflow as tf
import numpy as np
import argparse
import datetime
import logging
import pathlib
import sys
import os


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
            if (param.max_line_cnt is not None and
                line_cnt >= param.max_line_cnt):
                logger.info('Reach maximum line count %d' %
                                  param.max_line_cnt)
                break
            line_cnt += 1

            tokens = [BOS]
            tokens.extend(line.strip().split())
            tokens.append(EOS)

            if len(tokens) > param.max_sentence_length:
                continue

            # extra PAD at the end for lstm prediction
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
                    token = '<'+token+'>'
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
        embedding[param.dico[token]] = np.random.uniform(
            low=-0.25, high=0.25, size=param.emb_dim)

    logger.info('Finish loading pretrained embedding')
    return tf.convert_to_tensor(embedding, dtype=tf.float32)


def batch_generator(data, param, shuffle=True, batch_size=None):
    '''Batch generator for one epoch

    Args:
        data: numpy array
        param: experiment parameters
        shuffle: whether to shuffle data
    '''

    n = data.shape[0]
    if batch_size is None:
        batch_size = param.batch_size
    n_batch = (n-1)//batch_size+1

    # shuffle the data
    shuffle_idx = np.arange(n)
    if shuffle:
        shuffle_idx = np.random.permutation(shuffle_idx)
    shuffled_data = data[shuffle_idx]

    for batch in range(n_batch):
        start = batch * batch_size
        end = min((batch+1) * batch_size, n)
        yield shuffled_data[start:end], start


class RNNCell(object):

    def __init__(self, emb_dim, state_dim):

        self.emb_dim = emb_dim
        self.state_dim = state_dim
        with tf.name_scope('rnn_cell'):
            self.W = tf.get_variable('rnn_W',
                [emb_dim + state_dim, state_dim], tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.get_variable('rnn_b', [state_dim],
                tf.float32, initializer=tf.zeros_initializer())

    def __call__(self, x, state):

        features = tf.concat([x, state], axis=1)
        return tf.nn.relu(tf.matmul(features, self.W) + self.b)


class LSTMCell(object):

    def __init__(self, emb_dim, state_dim):

        self.emb_dim = emb_dim
        self.state_dim = state_dim
        with tf.name_scope('lstm_cell'):
            self.W = tf.get_variable('lstm_W',
                [emb_dim + state_dim, 4 * state_dim], tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.get_variable('lstm_b', [4 * state_dim],
                tf.float32, initializer=tf.zeros_initializer())

    def __call__(self, x, state):

        last_cell, last_hidden = state

        features = tf.concat([x, last_hidden], axis=1)
        features = tf.matmul(features, self.W) + self.b

        sd = self.state_dim
        forget_mask = tf.nn.sigmoid(features[:, :sd])
        input_mask = tf.nn.sigmoid(features[:, sd:2*sd])
        update_value = tf.nn.tanh(features[:, 2*sd:3*sd])
        output_mask = tf.nn.sigmoid(features[:, 3*sd:])

        cell = last_cell * forget_mask + input_mask * update_value
        hidden = tf.nn.tanh(cell) * output_mask
        return cell, hidden


class RNN(object):

    def __init__(self, param):

        self.param = param
        emb_dim, state_dim, num_steps, vocab_size = (param.emb_dim,
            param.state_dim, param.max_sentence_length, param.dico_size)

        with tf.name_scope('tokens'):
            self.input_x = tf.placeholder(
                tf.int32, [None, num_steps], name='input_x')
            self.input_y = tf.placeholder(
                tf.int32, [None, num_steps], name='input_y')
            self.batch_size = tf.shape(self.input_x)[0]

        with tf.name_scope('embedding'):
            if param.pretrained is None:
                self.embeddingW = tf.get_variable('embeddingW',
                    [vocab_size, emb_dim], tf.float32,
                    tf.contrib.layers.xavier_initializer())
            else:
                self.embeddingW = tf.convert_to_tensor(
                    load_pretrained_embeddings(param.pretrained, param),
                    tf.float32, 'embeddingW')

            embedding = tf.transpose(
                tf.nn.embedding_lookup(self.embeddingW, self.input_x),
                [1, 0, 2])

        with tf.name_scope('rnn'):
            self.h0 = tf.constant(0, tf.float32, [1, state_dim])
            self.rnn_cell = LSTMCell(emb_dim, state_dim)

            state = (tf.tile(self.h0, [self.batch_size, 1]),
                     tf.tile(self.h0, [self.batch_size, 1]))
            states = []
            for i in range(num_steps):
                state = self.rnn_cell(embedding[i], state)
                states.append(state[1])
            hidden = tf.transpose(tf.stack(states, 0), [1, 0, 2])

            flat_hidden = tf.reshape(hidden, [-1, state_dim])
            projected_dim = state_dim

        if param.hidden_proj_dim is not None:
            with tf.name_scope('project'):
                projected_dim = param.hidden_proj_dim
                self.proj_W = tf.get_variable('W_proj',
                    [state_dim, projected_dim], tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
                flat_hidden = tf.matmul(flat_hidden, self.proj_W)

        with tf.name_scope('softmax'):
            self.W_softmax = tf.get_variable('W_softmax',
                [projected_dim, vocab_size], tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            self.b_softmax = tf.get_variable('b_softmax', [vocab_size],
                tf.float32, initializer=tf.zeros_initializer())

            flat_logits = tf.matmul(
                flat_hidden, self.W_softmax) + self.b_softmax
            self.logits = tf.reshape(
                flat_logits, [self.batch_size, num_steps, vocab_size])

            self.predictions = tf.argmax(
                self.logits, axis=2, output_type=tf.int32)
            log_probability = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(log_probability)

        with tf.name_scope('evaluation'):
            mask = tf.cast(tf.not_equal(
                param.dico[PAD], self.input_y), tf.float32)
            masked_log_probability = mask * log_probability
            self.perplexity = tf.exp(tf.reduce_sum(masked_log_probability,
                axis=1) / tf.reduce_sum(mask, axis=1))

        with tf.name_scope('accuracy'):
            correct_mask = tf.cast(tf.equal(self.predictions, self.input_y),
                tf.float32) * mask
            self.accuracy = tf.reduce_mean(
                tf.reduce_sum(correct_mask, axis=1) /
                tf.reduce_sum(mask, axis=1))

    def predict(self, sentence):

        num_steps = self.param.max_sentence_length
        start_generation = tf.convert_to_tensor(False)

        state = (self.h0, self.h0)
        output = [tf.constant([0])] # sentinel
        for i in range(num_steps):
            start_generation = tf.logical_or(start_generation,
                tf.equal(sentence[0, i], self.param.dico[EOS]))
            feed_token = tf.cond(start_generation,
                lambda: output[-1], lambda: sentence[:, i])
            if i > 0:
                output[-1] = feed_token

            embedding = tf.nn.embedding_lookup(self.embeddingW, feed_token)
            embedding = tf.reshape(embedding, [1, self.param.emb_dim])

            state = self.rnn_cell(embedding, state)
            hidden = state[1]

            if self.param.hidden_proj_dim is not None:
                hidden = tf.matmul(hidden, self.proj_W)
            logits = tf.matmul(hidden, self.W_softmax) + self.b_softmax
            output.append(tf.argmax(logits, axis=1, output_type=tf.int32))

        return tf.stack(output[1:], axis=1)


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
    parser.add_argument('--conti_corpus', type=str, default=None,
                        help='Path to sentence continuation corpus')
    parser.add_argument('--max_line_cnt', type=int, default=None,
                        help='Maximum number of lines to load, default None')
    parser.add_argument('--max_sentence_length', type=int, default=30,
                        help='Maximum sentence length in consider, default 30')
    parser.add_argument('--max_conti_length', type=int, default=20,
                        help='Maximum sentence length to generate, default 20')
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
    assert os.path.exists(param.train_corpus)
    assert os.path.exists(param.eval_corpus)
    if param.conti_corpus is not None:
        assert os.path.exists(param.conti_corpus)
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

    # inverse dictionary
    logger.info('Building inverse dictionary')
    inverse_dico = {v: k for k, v in dico.items()}

    # transform corpus
    train_corpus = transform_corpus(train_corpus, dico, param)
    eval_corpus = transform_corpus(eval_corpus, dico, param)

    with tf.Session() as sess:
        rnn = RNN(param)

        # train_op
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer()
        grads_and_vars = optimizer.compute_gradients(rnn.loss)
        tgrads, tvars = zip(*grads_and_vars)
        tgrads, _ = tf.clip_by_global_norm(tgrads, param.max_grad_norm)
        grads_and_vars = zip(tgrads, tvars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

        # initialization
        sess.run(tf.global_variables_initializer())

        # train step
        def train_step(batch):
            feed_dict = {
                rnn.input_x: batch[:, :-1],
                rnn.input_y: batch[:, 1:]
            }
            _, step, loss, accuracy, predictions = sess.run(
                [train_op, global_step, rnn.loss, rnn.accuracy,
                rnn.predictions], feed_dict)
            if step % 20 == 0:
                logger.info('step %d, loss %f, acc %f' %
                    (step, loss, accuracy))

            return step, loss, accuracy, predictions

        # evaluation step
        def eval_step(batch):
            feed_dict = {
                rnn.input_x: batch[:, :-1],
                rnn.input_y: batch[:, 1:]
            }
            perplexity, accuracy, predictions = sess.run(
                [rnn.perplexity, rnn.accuracy, rnn.predictions],
                feed_dict)

            return perplexity, accuracy, predictions

        # training & evaluation
        logger.info('Start training')
        for epoch in range(param.n_epoch):
            logger.info('Start of epoch %d' % (epoch))
            for batch, idx in batch_generator(train_corpus, param):
                train_step(batch)

            # evaluation
            logger.info('Start evaluation for epoch %d' % (epoch))
            perplexity = np.array([], dtype=np.float)
            accuracy = 0
            predictions = None
            for batch, idx in batch_generator(eval_corpus, param, shuffle=False):
                ppl, acc, pred = eval_step(batch)

                # aggregate evaluation result
                perplexity = np.concatenate((perplexity, ppl))
                accuracy += acc * len(batch)
                predictions = (np.concatenate((predictions, pred), axis=0)
                    if predictions is not None else pred)

                if idx % (10 * param.batch_size) == 0:
                    logger.info('Finish iter %d of evaluation' % (idx//param.batch_size))
            accuracy = accuracy / len(eval_corpus)

            logger.info('Evaluation acc: %f' % accuracy)
            logger.info('Average perplexity: %f' % np.mean(perplexity))
            np.savetxt('/'.join([param.exp_path, 'eval.ckpt%d.perplexity' % epoch]),
                perplexity, fmt='%.10f')

            # save predicted sentences
            with open('/'.join([param.exp_path, 'eval.ckpt%d.prediction'
                % epoch]), 'w') as f:
                for prediction in predictions:
                    f.write(' '.join([inverse_dico[k] for k in prediction])+'\n')

        # generation
        if param.conti_corpus is not None:
            logger.info('Sentence continuation start')

            # load and transform corpus
            conti_corpus = load_corpus(param.conti_corpus, param)
            conti_corpus = transform_corpus(conti_corpus,
                dico, param)[:, :-1] # remove extra PAD

            def generation_step(batch):
                sentence = sess.run(rnn.predict(batch))
                return sentence

            logger.info('Predicting and writing to file')
            with open('/'.join([param.exp_path, 'continuation.txt']), 'w') as f:
                for batch, idx in batch_generator(conti_corpus, param,
                    shuffle=False, batch_size=1):
                    sentence = generation_step(batch)[0]
                    sentence = list(map(lambda x: inverse_dico[x], sentence))
                    for i in range(len(sentence)):
                        if sentence[i] == EOS:
                            sentence = sentence[:i+1]
                            break
                    sentence = sentence[:param.max_conti_length]
                    sentence = ' '.join(sentence)
                    f.write(sentence+'\n')

                    if idx % 10 == 0:
                        logger.info('Finish iter %d of generation' % (idx//param.batch_size))


if __name__ == '__main__':
    main()
