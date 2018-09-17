"""Beam search module.

Beam search takes the top K results from the model, predicts the K results for
each of the previous K result, getting K*K results. Pick the top K results from
K*K results, and start over again until certain number of results are fully
decoded.
"""

from six.moves import xrange
import numpy as np
import dynet as dy
import math


class Hypothesis(object):
    """Defines a hypothesis during beam search."""

    def __init__(self, tokens, prob, state, score=None):
        """Hypothesis constructor.

        Args:
          tokens: start tokens for decoding.
          prob: prob of the start tokens, usually 1.
          state: decoder initial states.
          score: decoder intial score.
        """
        self.tokens = tokens
        self.prob = prob
        self.state = state
        if score is None:
            self.score = math.log(prob[-1])
        else:
            self.score = score

    def extend_(self, token, prob, new_state):
        """Extend the hypothesis with result from latest step.

        Args:
          token: latest token from decoding.
          prob: prob of the latest decoded tokens.
          new_state: decoder output state. Fed to the decoder for next step.
        Returns:
          New Hypothesis with the results from latest step.
        """
        return Hypothesis(self.tokens+[token], self.prob+[prob],
                            new_state, self.score + math.log(prob))

    @property
    def latest_token(self):
        return self.tokens[-1]

    def __str__(self):
        return ('Hypothesis(prob = %.4f, tokens = %s)' % (self.prob,
                                                              self.tokens))


class BeamSearch(object):
    """Beam search."""

    def __init__(self, beam_size, start_token, end_token, max_steps):
        """Creates BeamSearch object.

        Args:
          beam_size: int.
          start_token: int, id of the token to start decoding with
          end_token: int, id of the token that completes an hypothesis
          max_steps: int, upper limit on the size of the hypothesis
        """
        self._beam_size = beam_size
        self._start_token = start_token
        self._end_token = end_token
        self._max_steps = max_steps

    def beam_search(self, encode_outputs, label_embeddings,
                        dec_lstm, w1, v1, w2, v2, w, b):
        """Performs beam search for decoding.

         Args:
            encode_outputs:
            label_embeddings:
            dec_lstm:
            w1, v1, w2, v2, w, b:
         Returns:
            hyps: list of Hypothesis, the best hypotheses found by beam search,
                    ordered by score
         """

        hyps_per_sentence = []
        #iterate over words in seq
        encode_outputs_col = dy.concatenate_cols(encode_outputs)
        query = dy.transpose(dy.rectify((w1 * encode_outputs_col) + v1))
        for encode_output in encode_outputs:
            encode_output_zeros = dy.zeros(encode_output.dim()[0])
            intial_state = dec_lstm.initial_state(
                                [encode_output, encode_output_zeros])
            complete_hyps = []
            hyps = [Hypothesis([self._start_token], [1.0], intial_state)]
            for steps in xrange(self._max_steps):
                if hyps != []:
                    # Extend each hypothesis.
                    # The first step takes the best K results from first hyps.
                    # Following steps take the best K results from K*K hyps.
                    all_hyps = []
                    for hyp in hyps:
                        label_embedding = label_embeddings[hyp.latest_token]
                        new_state = hyp.state.add_input(label_embedding)
                        decode_output = new_state.output()
                        key = dy.rectify((w2 * decode_output) + v2)
                        alpha = dy.softmax(query * key)
                        context = encode_outputs_col * alpha
                        label_probs = dy.softmax((w * dy.concatenate(
                                            [decode_output, context])) + b)
                        probs = label_probs.npvalue()
                        top_ids = np.argsort(probs)[-self._beam_size:]
                        top_probs = probs[top_ids]
                        all_hyps.extend([hyp.extend_(idx, prob, new_state)
                                    for idx, prob in zip(top_ids, top_probs)])
                    hyps = []

                    for h in self.best_hyps(self.sort_hyps(all_hyps)):
                        # Filter and collect any hypotheses that have the end token.
                        if h.latest_token == self._end_token and len(h.tokens)>2:
                            # Pull the hypothesis off the beam
                            #if the end token is reached.
                            complete_hyps.append(h)
                        elif h.latest_token == self._end_token:
                            pass
                        elif len(complete_hyps) >= self._beam_size \
                            and h.score < min(complete_hyps, key=lambda h: h.score).score:
                            pass
                        else:
                            # Otherwise continue to the extend the hypothesis.
                            hyps.append(h)
            hyps_per_word = self.best_hyps(self.sort_hyps(complete_hyps))
            hyps_per_sentence.append([(h.tokens[1:-1], h.score) for h in hyps_per_word])
        return hyps_per_sentence

    def sort_hyps(self, hyps):
        """Sort the hyps based on probs.
        Args:
          hyps: A list of hypothesis.
        Returns:
          hyps: A list of sorted hypothesis from highest prob to lowest
        """
        return sorted(hyps, key=lambda h: h.score, reverse=True)

    def best_hyps(self, hyp_sort):
        """return top <beam_size> hyps.

        Args:
          hyp_sort: A list of sorted hypothesis.
        Returns:
          hyps: A sub list of top <beam_size> hyps.
        """
        return hyp_sort[:self._beam_size]
