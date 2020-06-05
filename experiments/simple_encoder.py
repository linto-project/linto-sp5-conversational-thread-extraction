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
from allennlp.models import Model


from allennlp.modules import Seq2VecEncoder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from allennlp.data.vocabulary import Vocabulary


from overrides import overrides

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


from dummy_chat_reader import ChatReader, SimpleChatReader




class TurnMeanEmbedding(Seq2VecEncoder):
    pass

class ChatClassification(Model):


    def __init__(self, vocab, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder):
        super(ChatClassification, self).__init__(vocab)
        self.turn_encoder = embedder
        self.chat_encoder = encoder
        self.classif_layer = torch.nn.Linear(in_features=self.chat_encoder.hidden_size,
                                             out_features=2)
        self.accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        
    def forward(self,
                sentence,
                label = None):
        
        #breakpoint()
        mask = get_text_field_mask(sentence)
        
        turns_embeddings = self.turn_encoder(sentence)
        logging.info("forward pass: turn encodings done")
        final_h = self.chat_encoder(turns_embeddings,mask)
        logits = self.classif_layer(final_h)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self.accuracy(logits, label)

        return output_dict


    # expected by trainer ??
    #def get_parameters_for_histogram_tensorboard_logging(self,*args,**kwargs):
    #    return []
    

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
    
    token_indexers = {"tokens": SingleIdTokenIndexer()}
    
    tokenizer_cfg = Params({"word_splitter": {"language": "en"}})

    tokenizer = Tokenizer.from_params(tokenizer_cfg)
                                
    
    reader = SimpleChatReader(
        tokenizer=tokenizer,
        token_indexers=token_indexers,
        )
    train_instances = reader.read("./train_dummy.tsv")

    encoder_cfg = Params({'input_size': 100, 'hidden_size': 50, 'num_layers': 1,
                      'dropout': 0.25, 'bidirectional': True
    })
    encoder_cfg["type"] = "gru"
    encoder = Seq2VecEncoder.from_params(encoder_cfg)
    encoder.hidden_size = encoder_cfg["hidden_size"]


    
    logging.info("Vocabulary Creation")
    vocab = Vocabulary.from_instances(train_instances)

    
    glove_text_field_embedder = Embedding.from_params(vocab,Params({"pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
                                                              "embedding_dim": 100,
                                                              "trainable": False
    }))

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=100)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})


    
    #text_field_embedder= TextFieldEmbedder.from_params(text_field_embedder_cfg,vocab=vocab)
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


    model = ChatClassification(vocab,word_embeddings,encoder)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    iterator = BucketIterator(batch_size=1,sorting_keys=["tokens","num_tokens"])
    iterator.index_with(vocab)
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

                                                       
