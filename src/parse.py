import functools
import collections

import dynet as dy
import numpy as np

import trees
from beam.search import BeamSearch
from astar.search import astar_search

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"

def augment(scores, oracle_index):
    assert isinstance(scores, dy.Expression)
    shape = scores.dim()[0]
    assert len(shape) == 1
    increment = np.ones(shape)
    increment[oracle_index] = 0
    return scores + dy.inputVector(increment)

class Feedforward(object):
    def __init__(self, model, input_dim, hidden_dims, output_dim):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Feedforward")

        self.weights = []
        self.biases = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for prev_dim, next_dim in zip(dims, dims[1:]):
            self.weights.append(self.model.add_parameters((next_dim, prev_dim)))
            self.biases.append(self.model.add_parameters(next_dim))

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, x):
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            weight = dy.parameter(weight)
            bias = dy.parameter(bias)
            x = dy.affine_transform([bias, weight, x])
            if i < len(self.weights) - 1:
                x = dy.rectify(x)
        return x

class TopDownParser(object):
    def __init__(
            self,
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            tag_embedding_dim,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            label_hidden_dim,
            split_hidden_dim,
            dropout,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Parser")
        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim

        self.tag_embeddings = self.model.add_lookup_parameters(
            (tag_vocab.size, tag_embedding_dim))
        self.word_embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        self.lstm = dy.BiRNNBuilder(
            lstm_layers,
            tag_embedding_dim + word_embedding_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.f_label = Feedforward(
            self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size)
        self.f_split = Feedforward(
            self.model, 2 * lstm_dim, [split_hidden_dim], 1)

        self.dropout = dropout

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def parse(self, sentence, gold=None, explore=True):
        is_train = gold is not None

        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()

        embeddings = []
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            embeddings.append(dy.concatenate([tag_embedding, word_embedding]))

        lstm_outputs = self.lstm.transduce(embeddings)

        @functools.lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            forward = (
                lstm_outputs[right][:self.lstm_dim] -
                lstm_outputs[left][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][self.lstm_dim:] -
                lstm_outputs[right + 1][self.lstm_dim:])
            return dy.concatenate([forward, backward])

        def helper(left, right):
            assert 0 <= left < right <= len(sentence)

            label_scores = self.f_label(get_span_encoding(left, right))

            if is_train:
                oracle_label = gold.oracle_label(left, right)
                oracle_label_index = self.label_vocab.index(oracle_label)
                label_scores = augment(label_scores, oracle_label_index)

            label_scores_np = label_scores.npvalue()
            argmax_label_index = int(
                label_scores_np.argmax() if right - left < len(sentence) else
                label_scores_np[1:].argmax() + 1)
            argmax_label = self.label_vocab.value(argmax_label_index)

            if is_train:
                label = argmax_label if explore else oracle_label
                label_loss = (
                    label_scores[argmax_label_index] -
                    label_scores[oracle_label_index]
                    if argmax_label != oracle_label else dy.zeros(1))
            else:
                label = argmax_label
                label_loss = label_scores[argmax_label_index]

            if right - left == 1:
                tag, word = sentence[left]
                tree = trees.LeafParseNode(left, tag, word)
                if label:
                    tree = trees.InternalParseNode(label, [tree])
                return [tree], label_loss

            left_encodings = []
            right_encodings = []
            for split in range(left + 1, right):
                left_encodings.append(get_span_encoding(left, split))
                right_encodings.append(get_span_encoding(split, right))
            left_scores = self.f_split(dy.concatenate_to_batch(left_encodings))
            right_scores = self.f_split(dy.concatenate_to_batch(right_encodings))
            split_scores = left_scores + right_scores
            split_scores = dy.reshape(split_scores, (len(left_encodings),))

            if is_train:
                oracle_splits = gold.oracle_splits(left, right)
                oracle_split = min(oracle_splits)
                oracle_split_index = oracle_split - (left + 1)
                split_scores = augment(split_scores, oracle_split_index)

            split_scores_np = split_scores.npvalue()
            argmax_split_index = int(split_scores_np.argmax())
            argmax_split = argmax_split_index + (left + 1)

            if is_train:
                split = argmax_split if explore else oracle_split
                split_loss = (
                    split_scores[argmax_split_index] -
                    split_scores[oracle_split_index]
                    if argmax_split != oracle_split else dy.zeros(1))
            else:
                split = argmax_split
                split_loss = split_scores[argmax_split_index]

            left_trees, left_loss = helper(left, split)
            right_trees, right_loss = helper(split, right)

            children = left_trees + right_trees
            if label:
                children = [trees.InternalParseNode(label, children)]

            return children, label_loss + split_loss + left_loss + right_loss

        children, loss = helper(0, len(sentence))
        assert len(children) == 1
        tree = children[0]
        if is_train and not explore:
            assert gold.convert().linearize() == tree.convert().linearize()
        return tree, loss

class ChartParser(object):
    def __init__(
            self,
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            tag_embedding_dim,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            label_hidden_dim,
            dropout,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Parser")
        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim

        self.tag_embeddings = self.model.add_lookup_parameters(
            (tag_vocab.size, tag_embedding_dim))
        self.word_embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        self.lstm = dy.BiRNNBuilder(
            lstm_layers,
            tag_embedding_dim + word_embedding_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.f_label = Feedforward(
            self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size - 1)

        self.dropout = dropout

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def parse(self, sentence, gold=None):
        is_train = gold is not None

        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()

        embeddings = []
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            embeddings.append(dy.concatenate([tag_embedding, word_embedding]))

        lstm_outputs = self.lstm.transduce(embeddings)

        @functools.lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            forward = (
                lstm_outputs[right][:self.lstm_dim] -
                lstm_outputs[left][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][self.lstm_dim:] -
                lstm_outputs[right + 1][self.lstm_dim:])
            return dy.concatenate([forward, backward])

        @functools.lru_cache(maxsize=None)
        def get_label_scores(left, right):
            non_empty_label_scores = self.f_label(get_span_encoding(left, right))
            return dy.concatenate([dy.zeros(1), non_empty_label_scores])

        def helper(force_gold):
            if force_gold:
                assert is_train

            chart = {}

            for length in range(1, len(sentence) + 1):
                for left in range(0, len(sentence) + 1 - length):
                    right = left + length

                    label_scores = get_label_scores(left, right)

                    if is_train:
                        oracle_label = gold.oracle_label(left, right)
                        oracle_label_index = self.label_vocab.index(oracle_label)

                    if force_gold:
                        label = oracle_label
                        label_score = label_scores[oracle_label_index]
                    else:
                        if is_train:
                            label_scores = augment(label_scores, oracle_label_index)
                        label_scores_np = label_scores.npvalue()
                        argmax_label_index = int(
                            label_scores_np.argmax() if length < len(sentence) else
                            label_scores_np[1:].argmax() + 1)
                        argmax_label = self.label_vocab.value(argmax_label_index)
                        label = argmax_label
                        label_score = label_scores[argmax_label_index]

                    if length == 1:
                        tag, word = sentence[left]
                        tree = trees.LeafParseNode(left, tag, word)
                        if label:
                            tree = trees.InternalParseNode(label, [tree])
                        chart[left, right] = [tree], label_score
                        continue

                    if force_gold:
                        oracle_splits = gold.oracle_splits(left, right)
                        oracle_split = min(oracle_splits)
                        best_split = oracle_split
                    else:
                        best_split = max(
                            range(left + 1, right),
                            key=lambda split:
                                chart[left, split][1].value() +
                                chart[split, right][1].value())

                    left_trees, left_score = chart[left, best_split]
                    right_trees, right_score = chart[best_split, right]

                    children = left_trees + right_trees
                    if label:
                        children = [trees.InternalParseNode(label, children)]

                    chart[left, right] = (
                        children, label_score + left_score + right_score)

            children, score = chart[0, len(sentence)]
            assert len(children) == 1
            return children[0], score

        tree, score = helper(False)
        if is_train:
            oracle_tree, oracle_score = helper(True)
            assert oracle_tree.convert().linearize() == gold.convert().linearize()
            correct = tree.convert().linearize() == gold.convert().linearize()
            loss = dy.zeros(1) if correct else score - oracle_score
            return tree, loss
        else:
            return tree, score

class MyParser(object):
    def __init__(
            self,
            model,
            tag_vocab,
            word_vocab,
            char_vocab,
            label_vocab,
            tag_embedding_dim,
            word_embedding_dim,
            char_embedding_dim,
            label_embedding_dim,
            lstm_layers,
            lstm_dim,
            char_lstm_dim,
            dec_lstm_dim,
            attention_dim,
            label_hidden_dim,
            dropout,
            keep_valence_value,
            dropouts,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")
        self.model = model.add_subcollection("Parser")
        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab
        self.keep_valence_value = keep_valence_value
        self.lstm_dim = lstm_dim

        self.tag_embeddings = self.model.add_lookup_parameters(
            (tag_vocab.size, tag_embedding_dim))
        self.word_embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        embedding_dim = tag_embedding_dim + word_embedding_dim + char_lstm_dim
        self.enc_lstm = dy.BiRNNBuilder(
            lstm_layers,
            embedding_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.char_embeddings = self.model.add_lookup_parameters(
            (char_vocab.size, char_embedding_dim))

        self.char_lstm = dy.LSTMBuilder(
            1,
            char_embedding_dim,
            char_lstm_dim,
            self.model)

        self.label_embeddings = self.model.add_lookup_parameters(
            (label_vocab.size, label_embedding_dim))

        self.dec_lstm = dy.LSTMBuilder(
            1,
            label_embedding_dim,
            dec_lstm_dim,
            self.model)

        enc_out_dim = embedding_dim + 2 * lstm_dim
        dec_attend_dim = enc_out_dim + dec_lstm_dim
        Weights = collections.namedtuple('Weights', 'name prev_dim next_dim')
        ws = []
        ws.append(Weights(name='query', prev_dim=enc_out_dim, next_dim=attention_dim))
        ws.append(Weights(name='c_dec', prev_dim=enc_out_dim, next_dim=dec_lstm_dim))
        ws.append(Weights(name='key', prev_dim=dec_lstm_dim, next_dim=attention_dim))
        ws.append(Weights(name='attention', prev_dim=dec_attend_dim, next_dim=label_hidden_dim))
        ws.append(Weights(name='probs', prev_dim=label_hidden_dim, next_dim=label_vocab.size))
        self.ws = {}
        for w in ws:
            weight = self.model.add_parameters((w.next_dim, w.prev_dim))
            bias = self.model.add_parameters((w.next_dim))
            self.ws[w.name] = (bias, weight)

        self.dropout = dropout
        self.dropouts = dropouts

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def parse(self, sentence, gold=None, is_dev=False, predict_parms=None):
        is_train = gold is not None
        use_dropout = is_train and not is_dev

        if use_dropout:
            dropouts = self.dropouts
            self.enc_lstm.set_dropout(dropouts[0])
            self.char_lstm.set_dropout(dropouts[1])
            self.dec_lstm.set_dropout(dropouts[2])
        else:
            self.enc_lstm.disable_dropout()
            self.dec_lstm.disable_dropout()
            self.char_lstm.disable_dropout()
            dropouts = [0.]*len(self.dropouts)

        embeddings = []
        char_lstm = self.char_lstm.initial_state()
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            chars_embedding = []
            tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
            tag_embedding = dy.dropout(tag_embedding, dropouts[3])
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
            for c in [START] + list(word) + [STOP]:
                char_embedding = self.char_embeddings[self.char_vocab.index(c)]
                char_embedding = dy.dropout(char_embedding, dropouts[4])
                chars_embedding.append(char_embedding)
            word_char_embedding = char_lstm.transduce(chars_embedding)[-1]
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            word_embedding = dy.dropout(word_embedding, dropouts[5])
            embeddings.append(dy.concatenate([tag_embedding, word_embedding, word_char_embedding]))
        lstm_outputs = self.enc_lstm.transduce(embeddings)

        encode_outputs_list = []
        for e, l in zip(embeddings[1:-1], lstm_outputs[1:-1]):
            encode_outputs_list.append(dy.concatenate([e, l]))

        if is_train:
            decode_inputs = [(START,) + tuple(leaf.labels) + (STOP,) for leaf in gold.leaves()]
            losses = []
            encode_outputs = dy.concatenate_cols(encode_outputs_list)

            q = dy.affine_transform([*self.ws['query'], encode_outputs])
            query = dy.transpose(dy.rectify(q))

            for encode_output, decode_input in zip(encode_outputs_list, decode_inputs):

                label_embedding = []
                for label in decode_input[:-1]:
                    e = self.label_embeddings[self.label_vocab.index(label)]
                    label_embedding.append(dy.dropout(e, dropouts[6]))

                c_dec = dy.affine_transform([*self.ws['c_dec'], encode_output])
                h_dec = dy.zeros(c_dec.dim()[0])
                decode_init = self.dec_lstm.initial_state([c_dec, h_dec])
                decode_output_list = decode_init.transduce(label_embedding)
                decode_output = dy.concatenate_cols(decode_output_list)

                k = dy.affine_transform([*self.ws['key'], decode_output])
                key = dy.rectify(k)
                alpha = dy.softmax(query * key)
                context = encode_outputs * alpha
                x = dy.concatenate([decode_output, context])
                a = dy.affine_transform([*self.ws['attention'], x])
                attention = dy.rectify(a)

                p = dy.affine_transform([*self.ws['probs'], attention])
                probs = dy.softmax(p)

                log_prob = []
                for i, label in enumerate(decode_input[1:]):
                    id = self.label_vocab.index(label)
                    log_prob.append(-dy.log(dy.pick(dy.pick(probs, id), i)))
                losses.extend(log_prob)

            return None, losses

        else:
            import cProfile
            profile = cProfile.Profile()

            start = self.label_vocab.index(START)
            stop = self.label_vocab.index(STOP)
            astar_parms = predict_parms['astar_parms']
            for beam_size in predict_parms['beam_parms']:
                hyps = BeamSearch(start, stop, beam_size).beam_search(
                                                            encode_outputs_list,
                                                            self.label_embeddings,
                                                            self.dec_lstm,
                                                            self.ws)

                grid = []
                for i, (leaf_hyps, leaf) in enumerate(zip(hyps, sentence)):
                    row = []
                    for hyp in leaf_hyps:
                        labels = np.array(self.label_vocab.values)[hyp[0]].tolist()
                        partial_tree = trees.LeafMyParseNode(i, *leaf).deserialize(labels)
                        if partial_tree is not None:
                            row.append((partial_tree, hyp[1]))
                    grid.append(row)

                nodes = astar_search(grid, self.keep_valence_value, astar_parms)
                if nodes != []:
                    return nodes[0].trees[0], None

            children = [trees.LeafMyParseNode(i, *leaf) for i,leaf in enumerate(sentence)]
            tree = trees.InternalMyParseNode('S', children)
            return tree, None
