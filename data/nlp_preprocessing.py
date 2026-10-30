import torch
import torch.nn as nn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        all_words = positive + negative
        vocabulary = sorted(list(set(word for sentence in all_words for word in sentence.split())))
        vocab_map = {}

        for k, v in enumerate(vocabulary):
            vocab_map[v] = k + 1.0
        
        result = []
        for sentence in all_words:
            lst = []
            for word in sentence.split():
                lst.append(torch.tensor(vocab_map[word]))
            result.append(torch.tensor(lst))

        return torch.nn.utils.rnn.pad_sequence(result, padding_value=0, batch_first=True)