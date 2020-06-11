import itertools
import logging
from typing import Dict, List, Iterable
import torch
import torch.optim as optim

#from allennlp.data.dataset_readers import SnliReader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
# to be checked: reads a text as a list of sentences
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.fields import Field
from allennlp.data.fields import LabelField
from allennlp.data.fields import TextField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.modules.seq2vec_encoders import BertPooler
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding


# should be a better way
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm
from allennlp.modules import Seq2VecEncoder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from allennlp.data.vocabulary import Vocabulary


from overrides import overrides

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


from chat_reader import ChatReader


# instance = Instance({'lines': lines_field, 'labels': target})

#The things you need to be careful about are when you call your TextFieldEmbedder you need to add num_wrapping_dims=1, and you need to wrap your Seq2Vec encoder with TimeDistributed(). These are both in your Model.forward() method. Let me know if you need more detail.

#Adding num_embedding_dims =1 works, it turns out I didn't need to TimeDistributed the encoder. 

# TODO: turn encoder, following this kind of embedding for turns 


# then use the rest of graph_parser config changing the text_field_embedder, and removing pos tags and label prediction, sth like: 
# must check how dependencies are encoded in that -> "dataset_reader":{"type":"semantic_dependencies"}


class TurnMeanEmbedding(Seq2VecEncoder):
    pass

class ChatClassification(Seq2VecEncoder):


    def __init__(self, vocab, pretrained_model: str="bert-base-uncased", requires_grad: bool = True):
        super(ChatClassification, self).__init__()
        self.vocab = vocab
        self.turn_pooler = BertPooler(pretrained_model,requires_grad,dropout=0.0)
        #self.turn_pooler = 
        self.chat_encoder = StackedBidirectionalLstm(hidden_size= 400,
                                                     input_size= 768,
                                                     num_layers= 1,
                                                     recurrent_dropout_probability=0.3,
                                                     use_highway=True)
        self.classif_layer = torch.nn.Linear(in_features=self.chat_encoder.hidden_size,
                                             out_features=2)
        self.accuracy = CategoricalAccuracy()

        
    def forward(self,
                lines,
                label = None):
        turns_embeddings = [self.turn_pooler(turn) for turn in lines]
        output_seq, final_h, final_c = self.chat_encoder(turns_embeddings)
        logits = self.classif_layer(final_h)
        output = {"tag_logits": logits}

        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = 0#torch.nn.CrossEntropyLoss(logits, label.squeeze(-1))

        return output

    # expected by trainer ??
    def get_parameters_for_histogram_tensorboard_logging(self,*args,**kwargs):
        return []
    

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}



#TextFieldEmbedder you need to add num_wrapping_dims=1

    

if __name__=="__main__":
    
    from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer, PretrainedTransformerTokenizer
    from allennlp.data.token_indexers import PretrainedTransformerIndexer
    from allennlp.data.vocabulary import Vocabulary
    from allennlp.training.trainer import Trainer
    from allennlp.data.iterators import BucketIterator
    from allennlp.common import Params
    
    # for bert pretrained model
    #tokenizer = PretrainedTransformerTokenizer(
    #    model_name="bert-base-uncased",
    #    do_lowercase=True,
    #    )
    #token_indexers = PretrainedTransformerIndexer(
    #    model_name="bert-base-uncased",
    #    do_lowercase=True
    #    )

    
    #token_indexer_cfg =  Params({"tokens": {
    #    "type": "single_id",
    #    "lowercase_tokens": "true"
    #}})
    #token_indexers = TokenIndexer.from_params(token_indexer_cfg.pop("tokens"))
    token_indexers = {"tokens": SingleIdTokenIndexer()}
    
    tokenizer_cfg = Params({"word_splitter": {"language": "en"}})

    tokenizer = Tokenizer.from_params(tokenizer_cfg)
                                
    
    reader = ChatReader(
        tokenizer=tokenizer,
        token_indexers=token_indexers,
        )
    train_instances = reader.read("./train_dummy.tsv")
    
    all_turns = list(itertools.chain(*(x["lines"] for x in train_instances)))

    logging.info("Vocabulary Creation")
    vocab = Vocabulary.from_instances(all_turns)


    text_field_embedder_cfg = Params({
    "tokens": {
        "type": "sequence_encoding",
        "embedding": {
            "embedding_dim": 100
        },
        "encoder": {
            "type": "gru",
            "input_size": 100,
            "hidden_size": 50,
            "num_layers": 2,
            "dropout": 0.25,
            "bidirectional": True
        }
    }
    })
    #text_field_embedder = TextFieldEmbedder.from_params(text_field_embedder_cfg,vocab=vocab)
    

    # token based -> maybe do average from this
    glove_text_field_embedder = Embedding.from_params(vocab,Params({"pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
                                                              "embedding_dim": 100,
                                                              "trainable": False
    }))
    
    #text_field_embedder= TextFieldEmbedder.from_params(text_field_embedder_cfg)
    # """You need to be sure that the TextFieldEmbedder is expecting the same thing that your DatasetReader is producing, but that happens in the configuration file, and we'll talk about it later."""
    
    
    trainer_cfg = Params({"iterator": {"type": "basic",
                                       "batch_size": 32
    },
                          "trainer": {
                              "optimizer": {
                                  "type": "adam"
                              },
                              "num_epochs": 3,
                              "patience": 10,
                              "cuda_device": -1
                          }
    })


    model = ChatClassification(vocab)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    iterator = BucketIterator(batch_size=1,sorting_keys=["lines"])

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_instances,
                      should_log_parameter_statistics = False
    )


    if False:
        i0 = train_instances[0]
        t0 = i0["lines"][0]
        all_turns = list(itertools.chain(*(x["lines"] for x in train_instances)))
        vocab = Vocabulary.from_instances(all_turns)

                                                       
