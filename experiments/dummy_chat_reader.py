import itertools
import logging
import torch
from typing import Dict, List, Iterable
import numpy as np


from allennlp.data.dataset_readers.dataset_reader import DatasetReader
# to be checked: reads a text as a list of sentences
from allennlp.data.dataset_readers import TextClassificationJsonReader

from allennlp.data.fields import Field
from allennlp.data.fields import LabelField
from allennlp.data.fields import TextField, ListField, ArrayField
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

@DatasetReader.register("chat_reader")
class ChatReader(DatasetReader):
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
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        #file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            #header = next(data_file)
            inst_tokens = []
            inst_deps = []
            idx = 0
            for line in data_file:
                if line.strip()=="":
                    yield self.text_to_instance(inst_tokens,inst_deps)
                    idx = 0
                    inst_tokens, inst_deps = [],[]
                else:
                    head, turn = line.strip().split('\t')
                    if self.turn_encoder:
                        inst_tokens.append(turn)
                    else:
                        inst_tokens.append(self._tokenizer.tokenize(turn))
                    inst_deps.append((idx,head))
                    idx = idx 
            yield self.text_to_instance(inst_tokens,inst_deps)
    
    @overrides
    def text_to_instance(
        self,  # type: ignore
        inst_tokens: Iterable,
        inst_deps: Iterable,
    ) -> Instance:
        if self.turn_encoder:# bypass word embeddings by directly encoding turns-as-tokens
            sentence_embeddings = np.array(self.turn_encoder.encode(inst_tokens))
            print("shape of sentence batch",sentence_embeddings.shape)
            instance = Instance({'tokens': ArrayField(sentence_embeddings), 'label': LabelField(label="0")})
            return instance
        # otherwise prepare for sequence of sequence encoding
        # roughly similar to what happens in TextClassifierJson datareader
        lines_field = ListField([TextField(tokenized_line,self._token_indexers) for tokenized_line in inst_tokens])
        # labels are head index for each turn; root is "root"
        target = ListField([LabelField(label=head) for idx,head in inst_deps])  # or simply LabelField
        #instance = Instance({'lines': lines_field, 'labels': target})
        # mock label for testing of model as classification of a chat ; force label = 0
        instance = Instance({'lines': lines_field, 'label': LabelField(label="0")})
        return instance

###### VERY IMPORTANT: the name fields have to be found in the forward method of the models
#
#
#  (from the tutorial with a model classifying papers)
#       "The first thing to notice are the inputs to the method. Remember the DatasetReader we implemented? It created Instances with
#fields named title, abstract, and label. That's where the names to forward come from - they have to match the names that we gave to the
# fields in our DatasetReader"


        
@DatasetReader.register("simple_chat_reader")
class SimpleChatReader(DatasetReader):
    """
    Reads a file for chat problems
    
    same as above, but put all turns in one "sentence", for testing purposes
     
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
        lazy: bool = False,
    ) -> None:
        super(SimpleChatReader, self).__init__(lazy)
        self._tokenizer = tokenizer 
        self._max_sequence_length = max_sequence_length
        self._token_indexers = token_indexers
        
    
            
    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        #file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            #header = next(data_file)
            inst_tokens = []
            inst_deps = []
            idx = 0
            for line in data_file:
                if line.strip()=="":# new chat begins, yield previous one
                    yield self.text_to_instance(inst_tokens,inst_deps)
                    idx = 0
                    inst_tokens, inst_deps = [],[]
                else:
                    head, turn = line.strip().split('\t')
                    inst_tokens.extend(self._tokenizer.tokenize(turn))
                    inst_deps.extend((idx,head))
                    idx = idx + 1
            yield self.text_to_instance(inst_tokens,inst_deps)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        inst_tokens: List[Token],
        inst_deps: Iterable,
    ) -> Instance:
        # roughly similar to what happens in TextClassifierJson datareader
        token_field = TextField(inst_tokens,self._token_indexers)
        # labels are head index for each turn; root is "root"
        #print("deps=",inst_deps,"$")
        #target = LabelField(label=[head for idx,head in inst_deps])
        #label_field = SequenceLabelField(labels=tags, sequence_field=token_field)
        #instance = Instance({'lines': lines_field, 'labels': target})
        # mock label for testing of model as classification of a chat ; force label = 0
        instance = Instance({'sentence': token_field, 'label': LabelField(label="0")})
        return instance




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
                                
    
    reader = SimpleChatReader(
        tokenizer=tokenizer,
        token_indexers=token_indexers,
        )
    train_instances = reader.read("./train_dummy.tsv")
    
    #vocab = Vocabulary.from_instances([x["sentence"] for x in train_instances])
    vocab = Vocabulary.from_instances(train_instances)



                                                       
