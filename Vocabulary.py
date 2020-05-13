class Vocabulary(object):
    """
    Class to process text and extract Vocabulary for mapping
    """

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        :param token_to_idx:-->int, a ore-existing map of tokens to indices
        :param add_unk: -->bool, a flag that indicates whether to add the UNK token
        :param unk_token: -->str, the UNK token to add into Vocabulary
        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = \
            {idx: token for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def to_serializeable(self):
        """
        returns a dictionary that can be serialized(序列化)
        """
        return {
            'token_to_inx': self._token_to_idx,
            'add_unk': self._add_unk,
            'unk_token': self._unk_token
        }

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):
        """
        Update mapping dicts based on the token
        :param token: -->str, the item to add into the Vocabulary
        :return: -->int, the integer corresponding to the token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def lookup_index(self, index):
        """
        return the token associated with the index
        :param index: -->int, the index to look up
        :return: -->str, the token corresponding to the index
        :raises: KeyError: if the index is not in the vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary " % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)
