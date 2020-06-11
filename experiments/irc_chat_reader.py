import itertools
import logging
import os
import torch
from typing import Dict, List, Iterable
import numpy as np


from allennlp.data.dataset_readers.dataset_reader import DatasetReader
# to be checked: reads a text as a list of sentences
from allennlp.data.dataset_readers import TextClassificationJsonReader

from allennlp.data.fields import Field
from allennlp.data.fields import LabelField
from allennlp.data.fields import TextField, ListField, ArrayField, SequenceLabelField, AdjacencyField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers import Tokenizer
from allennlp.modules.seq2vec_encoders import BertPooler
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder



from sentence_transformers import SentenceTransformer


from overrides import overrides

logger = logging.getLogger(__name__)


@TokenEmbedder.register("pass_through_dict")
class PassThroughTokenEmbedder(TokenEmbedder):
    """
    Assumes that the input is already vectorized in some way,
    and just returns it.

    Modified from pass_through to have a Dict[str,Tensor] as output instead so that it can be 
    accepted by Textfielembedder

    Registered as a `TokenEmbedder` with name "pass_through_dict".
    # Parameters
    hidden_dim : `int`, required.



    """

    def __init__(self, hidden_dim: int) -> None:
        self.hidden_dim = hidden_dim
        super().__init__()

    def get_output_dim(self):
        return self.hidden_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return {"tokens":tokens}

@DatasetReader.register("irc_chat_reader")
class IRCChatReader(DatasetReader):
    """
    Reads a file for chat problems
    
    just for testing encoding, not necessarily the right data format reader       
     
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``SpacyTokenizer()``)
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.

    TODO: 
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_sequence_length: int = None,
        encode_turns: bool = False,
        lazy: bool = False,
    ) -> None:
        super(ChatReader, self).__init__(lazy)
        self._tokenizer = tokenizer 
        self._max_sequence_length = max_sequence_length
        self._token_indexers = token_indexers
        self.encode_turns = encode_turns
        self.turn_encoder = SentenceTransformer('bert-base-nli-mean-tokens') if encode_turns else None
        
        
    @overrides
    def _read(self, dir_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        #file_path = cached_path(file_path)

        files_id = set([".".join(x.split(".")[0:2]) for x in os.listdir(dir_path)])

        for file_id in files_id:
            inst_tokens = []
            arcs = []
            idx = 0
            first_line = 5000

            annotation_filepath = os.path.join(dir_path, file_id + ".annotation.txt")
            with open(annotation_filepath, "r") as annotation_file:
                logger.info("Reading instances from lines in file at: %s", annotation_filepath)
                for line in annotation_file:
                    t, s, _ = line.split()
                    target, source = map(int, [t, s])
                    if first_line > min(target, source):
                        first_line = min(target, source)
                    arcs.append((source, target))

            inst_arcs = [(s - first_line, t - first_line) for s, t in arcs]
            # print(inst_arcs)

            chat_filepath = os.path.join(dir_path, file_id + ".tok.txt")
            with open(chat_filepath, "r") as chat_file:
                logger.info("Reading instances from lines in file at: %s", chat_filepath)
                for idx, line in enumerate(chat_file):
                    turn = line.strip()
                    if idx < first_line:
                        continue
                    # print(idx, turn)
                    if self.turn_encoder:
                        inst_tokens.append(turn)
                    else:
                        inst_tokens.append(self._tokenizer.tokenize(turn))

            yield self.text_to_instance(inst_tokens, inst_arcs)

        # with open(file_path, "r") as data_file:
        #     logger.info("Reading instances from lines in file at: %s", file_path)
        #     #header = next(data_file)
        #     inst_tokens = []
        #     inst_deps = []
        #     idx = 0
        #     for line in data_file:
        #         if line.strip()=="":
        #             yield self.text_to_instance(inst_tokens,inst_deps)
        #             idx = 0
        #             inst_tokens, inst_deps = [],[]
        #         else:
        #             head, turn = line.strip().split('\t')
        #             if self.turn_encoder:
        #                 inst_tokens.append(turn)
        #             else:
        #                 inst_tokens.append(self._tokenizer.tokenize(turn))
        #             inst_deps.append((idx,head))
        #             idx = idx 
        #     yield self.text_to_instance(inst_tokens,inst_deps)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        inst_tokens: Iterable,
        # inst_deps: Iterable,
        inst_arcs: Iterable,
    ) -> Instance:
        if self.turn_encoder:# bypass word embeddings by directly encoding turns-as-tokens
            sentence_embeddings = np.array(self.turn_encoder.encode(inst_tokens))
            print("shape of sentence batch",sentence_embeddings.shape)
            instance = Instance({'tokens': ArrayField(sentence_embeddings), 'label': LabelField(label="0")})
            return instance
        fields: Dict[str, Field] = {}
        # otherwise prepare for sequence of sequence encoding
        # roughly similar to what happens in TextClassifierJson datareader
        lines_field = ListField([TextField(tokenized_line, self._token_indexers) for tokenized_line in inst_tokens])
        fields["lines"] = lines_field
        # labels are head index for each turn; root is "root"
        # target = SequenceLabelField([head for idx, head in inst_deps],lines_field)  # or simply LabelField
        #instance = Instance({'lines': lines_field, 'labels': target})
        # mock label for testing of model as classification of a chat ; force label = 0
        fields["arcs"] = AdjacencyField(inst_arcs, lines_field)
        return Instance(fields)

###### VERY IMPORTANT: the name fields have to be found in the forward method of the models
#
#
#  (from the tutorial with a model classifying papers)
#       "The first thing to notice are the inputs to the method. Remember the DatasetReader we implemented? It created Instances with
#fields named title, abstract, and label. That's where the names to forward come from - they have to match the names that we gave to the
# fields in our DatasetReader"


if __name__=="__main__":
    from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer, PretrainedTransformerTokenizer
    from allennlp.data.token_indexers import PretrainedTransformerIndexer
    from allennlp.data.vocabulary import Vocabulary
    from allennlp.training.trainer import Trainer
    from allennlp.data.iterators import BucketIterator
    from allennlp.common import Params

    token_indexers = {"tokens": SingleIdTokenIndexer()}
    tokenizer_cfg = Params({"word_splitter": {"language": "en"}})
    tokenizer = Tokenizer.from_params(tokenizer_cfg)
    reader = IRCChatReader(
        tokenizer=tokenizer,
        token_indexers=token_indexers,
        )
    train_instances = reader.read("../../../data/irc-disentanglement/data/train")
    #vocab = Vocabulary.from_instances([x["sentence"] for x in train_instances])
    vocab = Vocabulary.from_instances(train_instances)


