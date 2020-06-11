from allennlp.models import Model
from typing import Dict, List, Iterable
from allennlp.modules import TimeDistributed
from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Seq2VecEncoder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
import torch


class HierarchicalChatClassification(Model):

    # complicated way of doing it: better to put turn_encoder inside a curtom textfield embedder
    def __init__(self, vocab, embedder: TextFieldEmbedder, turn_encoder: Seq2VecEncoder, chat_encoder: Seq2VecEncoder):
        super(HierarchicalChatClassification, self).__init__(vocab)
        # turn encoder has to distribute over turns of a chat instance
        self.turn_encoder = TimeDistributed(turn_encoder)
        self.chat_encoder = chat_encoder
        self.text_embedder = embedder
        self.classif_layer = torch.nn.Linear(in_features=self.chat_encoder.hidden_size,
                                             out_features=2)
        self.accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        
    def forward(self,
                lines,
                label = None):
        
        #breakpoint()
        # mask for each turn of each chat of the batch: shape = (batch_size x max_turns x tokens)
        mask = get_text_field_mask(lines,num_wrapping_dims=1)

        # chat turns fetching embedding
        # turns_embedding tensor is (batch_size x turns x max tokens x token embedding size)
        turns_embeddings = self.text_embedder(lines,num_wrapping_dims=1)
        
        # encoding turns
        # turn_h has shape (batch_size x turns x encoder_output_size) eg (1x3x50)
        turn_h = self.turn_encoder(turns_embeddings,mask)
        
        # encoding chat 
        # mask for chats is now nb of turns; beware weird return type of torch.max (tuple) 
        chat_mask = mask.max(axis=2)[0]
        # since chat encoder is seq2vec, output is just for one state and shape (batch_size x encoder_output_size) 
        chat_h = self.chat_encoder(turn_h,chat_mask)
        
        
        logits = self.classif_layer(chat_h)
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

