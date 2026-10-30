import torch
import torch.nn as nn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        vocabulary = set()
        combined = positive + negative
        for sentence in combined:
            for word in sentence.split():
                vocabulary.add(word)

        vocabulary = sorted(vocabulary)

        word2int = {}
        for i, word in enumerate(vocabulary):
            word2int[word] = i + 1

        def encode(sentence):
            integer_vocabulary = []
            for word in sentence.split():
                integer_vocabulary.append(word2int[word])
            return torch.tensor(integer_vocabulary)

        var_len_vocabulary = []
        for sentence in combined:
            var_len_vocabulary.append(encode(sentence))

        return nn.utils.rnn.pad_sequence(var_len_vocabulary, batch_first = True)