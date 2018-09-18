import functools

import dynet as dy
import numpy as np

import trees
from beam.search import BeamSearch
from astar.search import solve_tree_search

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
            label_vocab,
            tag_embedding_dim,
            word_embedding_dim,
            label_embedding_dim,
            lstm_layers,
            lstm_dim,
            label_hidden_dim,
            attention_dim,
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

        embedding_dim = tag_embedding_dim + word_embedding_dim
        self.enc_lstm = dy.BiRNNBuilder(
            lstm_layers,
            embedding_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.label_embeddings = self.model.add_lookup_parameters(
            (label_vocab.size, 2 * lstm_dim))

        self.dec_lstm = dy.LSTMBuilder(
            1,
            2 * lstm_dim,
            embedding_dim + 2 * lstm_dim,
            self.model)

        self.weights = []
        self.biases = []
        state_size = embedding_dim + 2 * lstm_dim
        for _ in range(2):
            self.weights.append(self.model.add_parameters((attention_dim, state_size)))
            self.biases.append(self.model.add_parameters((attention_dim)))
        self.weights.append(self.model.add_parameters((label_vocab.size, 2 * state_size)))
        self.biases.append(self.model.add_parameters((label_vocab.size)))

        self.dropout = dropout

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def parse(self, sentence, gold=None):
        is_train = gold is not None

        def affine(bias, weight, x, non_linearity=dy.rectify):
            x = dy.affine_transform([bias, weight, x])
            if non_linearity is None:
                return x
            return non_linearity(x)

        if is_train:
            self.enc_lstm.set_dropout(self.dropout)
            self.dec_lstm.set_dropout(self.dropout)
        else:
            self.enc_lstm.disable_dropout()
            self.dec_lstm.disable_dropout()

        embeddings = []
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            embeddings.append(dy.concatenate([tag_embedding, word_embedding]))
        lstm_outputs = self.enc_lstm.transduce(embeddings)

        encode_outputs = [dy.concatenate([e, l]) for e, l in zip(embeddings, lstm_outputs)]
        encode_outputs = encode_outputs[1:-1]

        ws = [(dy.parameter(b), dy.parameter(w)) for b, w in zip(self.biases, self.weights)]

        if is_train:
            decode_inputs = [(START,) + leaf.label + (STOP,) for leaf in gold.leaves()]
            losses = []
            _encode_outputs = dy.concatenate_cols(encode_outputs)
            query = dy.transpose(affine(*ws[0], _encode_outputs))
            for encode_output, decode_input in zip(encode_outputs, decode_inputs):

                label_embedding = [self.label_embeddings[self.label_vocab.index(label)]
                                        for label in decode_input[:-1]
                                        ]

                encode_output_zeros = dy.zeros(encode_output.dim()[0])
                intial_state = self.dec_lstm.initial_state([encode_output,
                                                            encode_output_zeros]
                                                            )
                decode_output = dy.concatenate_cols(intial_state.transduce(label_embedding))
                key = affine(*ws[1], decode_output)
                alpha = dy.softmax(query * key)
                context = _encode_outputs * alpha
                x = dy.concatenate([decode_output, context])
                #TODO x = affine(*ws[3], x)
                probs = affine(*ws[2], x, dy.softmax)
                log_prob = []
                for i, label in enumerate(decode_input[1:]):
                    id = self.label_vocab.index(label)
                    log_prob.append(-dy.log(dy.pick(dy.pick(probs, id), i)))
                losses.extend(log_prob)

            return None, losses

        else:
            bs = BeamSearch(5,
                            self.label_vocab.index(START),
                            self.label_vocab.index(STOP),
                            28)

            beams = bs.beam_search(encode_outputs, self.label_embeddings, self.dec_lstm, ws)

            def helper(beam, leaf, index=0):
                label = np.array(self.label_vocab.values)[beam[0]].tolist()[1:]
                children = [trees.LeafMyParseNode(index, *leaf)]
                while label:
                    if not (label[0].startswith(trees.R) or label[0].startswith(trees.L)):
                        p_label = label[0]
                        label = label[1:]
                    while label and (label[0].startswith(trees.R) or label[0].startswith(trees.L)):
                        index += 1
                        children.append(trees.MissMyParseNode(label[0], index))
                        label = label[1:]
                    try:
                        children = [trees.InternalMyParseNode(p_label, children)]
                    except:
                        import pdb; pdb.set_trace()
                return (children[-1], beam[1])

            beams = [[helper(b, word, i) for b in beam]
                        for i, (beam, word) in enumerate(zip(beams, sentence))]

            tree =  solve_tree_search(beams, True, 1, 100., 10., 0.2)
            return tree, None
