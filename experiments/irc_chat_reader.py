import logging
import os
import torch
from typing import Dict, List, Iterable
import numpy as np
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
# to be checked: reads a text as a list of sentences
from allennlp.data.fields import Field
from allennlp.data.fields import LabelField
from allennlp.data.fields import TextField, ListField, ArrayField, SequenceLabelField, AdjacencyField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers import Tokenizer
from allennlp.modules.seq2vec_encoders import BertPooler
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from overrides import overrides

logger = logging.getLogger(__name__)


#######################################
#### TODO: 
####  - add metadata to instances: 
####       x- source file
####       - original index in chat for each turn
####       x- speaker for each turn
####       x- addresse in a turn when present
####       - 
####  - preprocessing:
####       - remove sentence markeup (<s> </s>)
###        - replace user placeholder with sth that wont be split up, like <user> -> #user
###        - tokenization foireuse -> prendre raw direct ?
###        x- remove server messages ? (===) => "server"
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
    max_sequence_length: int, cut the chat after this nb of turns (starting at the first turn that is related to another one)
    lazy: cf DatasetReader
    raw: bool, read raw files instead of pre-tokenized files
    min_link: int, read only turns after this nb of turns
    sample: int,  limit to that nb of instances (for testing)
    clip: int,  clip chats to maximal nb of turns
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_sequence_length: int = None,
        lazy: bool = False,
        raw: bool = False,
        sub_sequence: int = None,
        min_link: int = 1000,
        sample: int = None,
        clip: int = None,  
    ) -> None:
        
        super(ChatReader, self).__init__(lazy)
        self._tokenizer = tokenizer 
        self._max_sequence_length = max_sequence_length
        self._token_indexers = token_indexers
        self.raw = raw
        self.sample = sample
        self.clip = clip
        self.min_link = min_link
        self.sub_sequence = sub_sequence
        
    @overrides
    def _read(self, dir_path: str) -> Iterable[Instance]:
        # TODO: 
        #    refactoring, to have in separate functions
        #        - reading of file, output list of turns and edges
        #        - preprocessing of turns
        #        - feature extraction, including chat-level (list of users, e.g.)
        #    here must just remain the wrapping of all this
        clip = self.clip
        files_id = set()
        for filename in os.listdir(dir_path):
            file_id = os.path.splitext(os.path.splitext(filename)[0])[0]
            files_id.add(file_id)

        for file_id in list(files_id):#[:self.sample]:
            inst_tokens = []
            arcs = []
            idx = 0
            first_line = 5000

            annotation_filepath = os.path.join(dir_path, file_id + ".annotation.txt")
            with open(annotation_filepath, "r") as annotation_file:
                logger.debug("Reading instances from lines in file at: %s"%annotation_filepath)
                for line in annotation_file:
                    # t, s, _ = line.split()
                    target, source = map(int, line.split()[0:2])
                    if target < self.min_link or source < self.min_link:
                        continue
                    if first_line > min(target, source):
                        first_line = min(target, source)
                    arcs.append((source, target))

            # if clip :
            #     first_line = max(first_line, 1000)

            inst_arcs = []
            # save = []
            # save2 = []
            for s, t in arcs:
                s = s - first_line
                t = t - first_line
                # save.append((s, t, first_line))
                # save2.append((s, t, clip))
                if clip:
                    if s < clip and t < clip and s >= 0 and t >= 0:
                        inst_arcs.append((s, t))
                else:
                    inst_arcs.append((s, t))
            # if inst_arcs == []:
            #     print(save)
            #     print(save2)
            #     print(annotation_filepath)
            #     sys.exit(1)

            suffix = ".raw.txt" if self.raw else ".tok.txt"
            chat_filepath = os.path.join(dir_path, file_id + suffix)
            all_turns = []
            # collect features here. could even be a dict of functions to call
            features = {"is_server":[],
                        "speaker":[],
                        "addressee":[]}
            
            with open(chat_filepath, "r") as chat_file:
                logger.debug("Reading instances from lines in file at: %s"%chat_filepath)
                #print("Reading instances from lines in file at: %s"%chat_filepath)
                for idx, line in enumerate(chat_file):
                    if clip is not None and (idx - first_line) >= clip:
                        break
                    if idx < first_line:# or (clip is not None and (idx-first_line)>=clip):
                        continue
                    turn = line.strip()
                    if self.raw:# remove timestamp
                        if not(turn.startswith("===")):
                            turn = turn.split("]",1)[1].strip()
                        turn = preprocess_raw_turn(turn)
                    all_turns.append(turn)
                    
                    features["speaker"].append(get_user(turn))
                    features["is_server"].append(is_server(turn))
                    
                    inst_tokens.append(self._tokenizer.tokenize(turn))
            
            all_speakers = set(features["speaker"])
            for turn in all_turns:
                addressee = get_addressee(turn,user_list=all_speakers)
                features["addressee"].append(addressee)
            
            for one_instance in self.text_to_instance(inst_tokens, inst_arcs,
                                        sub_sequence = self.sub_sequence,
                                        metadata={"file_source":file_id,
                                                  "first_line":first_line,
                                                  "features":features,
                                                  "tokens":[]}
                                       ):
                yield one_instance
            
            
            
    def target_short_sequence_instance(
         self,  # type: ignore
        inst_tokens: Iterable,
        inst_arcs: Iterable,
        context_length: int = 15,
        metadata = {"tokens":[]}
    ) -> Instance:
        """generate instances as short sequences (of given length) before a target turn
        with labels being dependency between target and its head;
        if head is out of context ? -> force previous turn ? other heuristics ?
        
        could also be used to refactor full-chat instances, by providing start="first" line, end=end of chat
        """
        all = []
        for i,turn in enumerate(inst_tokens):
            start = max(0,i-context_length)
            context_turns = inst_tokens[start:i+1]
            data = self.extract_data(inst_arcs, metadata,start,i)
            fields: Dict[str, Field] = {}
            seq_field = ListField([TextField(tokenized_line, self._token_indexers) for tokenized_line in context_turns])
            data["tokens"] = context_turns
            data["file_source"] = metadata["file_source"]
            fields["lines"] = seq_field
            try:
                fields["arcs"] = AdjacencyField(data["arcs"], seq_field)
            except:
                print("error at index",i,"interval should be (%d,%d)"%(start,i))
                breakpoint()
            # example additional features, should be a separate function
            features = data["features"]
            # does src address target ? 
            fields["rel_features"] = ArrayField(target_address_src_matrix(features["speaker"],features["addressee"]))
            # distance between src and target
            fields["offsets"] = ArrayField(turn_distance_matrix(context_turns),dtype=int)
            # is the turn the server ? 
            fields["is_server"] = ArrayField(np.array(features["is_server"]))
            # should be listfield too ? one for each line ? 
            fields["metadata"] = MetadataField(data)
            all.append(Instance(fields))
        return all
            
    def extract_data(self,arcs,metadata,start,end):
        """ get relevant data (meta + dependencies) for given sub-sequence (start,end); 
        reindex dependencies and features to start at "start" index
        """
        data = {}
        data["arcs"] = []
        data["features"] = {}
        # reindex at 0
        shift = start
        # (target,src) = (target turn, its head)
        for (t,s) in arcs: 
            #if (start<=s<=end and start<=t<=end):
            # keep only links to the target turn, and only with heads within the context
            if t==end and (start<=s<=end):
                data["arcs"].append((t-shift,s-shift))
            
                
        if len(data["arcs"])==0:
            # if no head for target within given context ? default head = self
            data["arcs"] = [(end-shift,end-shift)]
        for feature in ("speaker","addressee","is_server"):
            data["features"][feature] = metadata["features"][feature][start:end+1]
        return data
        

    
    # instance = whole chat
    def text_to_instance_whole(
        self,  # type: ignore
        inst_tokens: Iterable,
        # inst_deps: Iterable,
        inst_arcs: Iterable,
        metadata = {"tokens":[]}
    ) -> Instance:
        # TODO
        #    - integrate output of feature extraction, in separate fields
        #           . turn specific features as a vector of values
        #           . source->target specific features as a tensor of values 
        #                    - same speaker
        #                    - address == speaker
        #                    - offsets (nb of turns between)
        fields: Dict[str, Field] = {}
       
        lines_field = ListField([TextField(tokenized_line, self._token_indexers) for tokenized_line in inst_tokens])
        fields["lines"] = lines_field
        fields["arcs"] = AdjacencyField(inst_arcs, lines_field)
        # example additional features, should be a separate function
        features = metadata["features"]
        
        # does src address target ? 
        fields["rel_features"] = ArrayField(target_address_src_matrix(features["speaker"],features["addressee"]))
        # distance between src and target
        fields["offsets"] = ArrayField(turn_distance_matrix(inst_tokens),dtype=int)
        # is the turn the server ? 
        fields["is_server"] = ArrayField(np.array(features["is_server"]))
        
        # should be listfield too ? one for each line ? 
        fields["metadata"] = MetadataField(metadata)

        return [Instance(fields)]

    
    @overrides
    def text_to_instance(
        self,  # type: ignore
        inst_tokens: Iterable,
        inst_arcs: Iterable,
        sub_sequence:int = None,
        metadata = {"tokens":[]}
    ) -> Instance:
        if sub_sequence: 
            return self.target_short_sequence_instance(inst_tokens,inst_arcs,context_length=sub_sequence,metadata=metadata)
        else:
            return self.text_to_instance_whole(inst_tokens,inst_arcs,metadata=metadata)
    
###### VERY IMPORTANT: the name fields have to be found in the forward method of the models

##################
# preprocessing
##################
import re
# find the turn speaker
user_re = re.compile("(^(<|( *))(?P<user>[^\s>]+)(>|\s))")
# find if the speaker explicitely replies to someone
address_re = re.compile("^<(\w)+> (?P<address>[^\s]+?)\s*[,:]+")
# find website addresses
http_re = re.compile(r"http[s]*:[^\s]+")


default_cfg = {
    "strip_markup":False,
    "mark_server_turn":True,
    "mark_http":True,
    "normalize_nb":False,
    }

def preprocess_raw_turn(turn,cfg=default_cfg):
    for option in cfg:
        if cfg[option]:
            turn = eval(option)(turn)
    return turn


def strip_markup(turn):
    return turn.replace("<","").replace(">","")
    

def mark_server_turn(turn):
    if turn.strip().startswith("==="): 
        return turn.replace("===","_server") 
    else: 
        return turn

def mark_http(turn):
    return http_re.sub("#http",turn)

table = str.maketrans("0123456789","0000000000")
def normalize_nb(turn):
    return turn.translate(table)


#########################
# feature extraction
# mostly replicating Kummerfeld et al.
# cf https://github.com/jkkummerfeld/irc-disentanglement/blob/master/src/disentangle.py
##########################
import sys

def get_timestamp(turn):
    return None

def is_server(turn):
    return turn.startswith("_server") or turn.startswith("===")

def get_user(turn):
    if is_server(turn): return "server"
    a = user_re.match(turn)
    if a: 
        return a.groupdict()["user"]
    else: 
        print("not finding speaker -->",turn,file=sys.stderr)
        return "?"
# to be checked, but seems reasonable
def has_query(turn):
    return "?" in turn


def get_addressee(turn,user_list={}):
    """check if beginning of turn addresses a specific user, or is a user from a known list is mentioned"""
    # empty turn except for user id
    if is_server(turn) or turn.startswith("*"): return None
    if turn.split(">")[1]=="": return None
    if user_list!={}:
            possible = turn.split()[1].strip(",:")
            if possible in user_list: return possible
    else:
        a = address_re.match(turn)
        if a: return a.groupdict().get("address",None)
    return None

####### relational features ########
def same_speaker_matrix(speakers):
    dim=len(speakers)
    m = np.zeros(shape=(dim,dim))
    for i in range(dim):
        for j in range(dim):
            m[i,j] = (i<j) and speakers[i]==speakers[j]
    return m

def target_address_src_matrix(speakers,addressees):
    dim=len(speakers)
    m = np.zeros(shape=(dim,dim))
    for i in range(dim):
        for j in range(dim):
            m[i,j] = (i<j) and speakers[i]==addressees[j]
    return m

def turn_distance_matrix(inst_tokens):
    """just a matrix of index distance on all turns
    placeholder in case we want to do some pre-filtering/transformation
    """
    dim=len(inst_tokens)
    m = np.zeros(shape=(dim,dim))
    for i in range(dim):
        for j in range(dim):
            # arcs are src (j) -> target (i), with i>j 
            m[i,j] = i-j
    return m

def no_src_turn_between_matrix(instance,src_id,target_id):
    pass

# missing from kummerfeld
    # "?" in turn (either src/target) 

# List from Kummerfeld:
    # General information about this sample of data
    # Year
    # Number of messages per minute
    # Query (==src)
    #  - Normal message or system message
    #  - Hour of day
    #  - Is it targeted
    #  - Is there a previous message from this user?
    #  - Did the previous message from this user have a target?
    #  - How long ago was the previous message from this user in messages?
    #  - How long ago was the previous message from this user in minutes?
    #  - Are they a bot?
    # Link (==target)
    #  - Normal message or system message
    #  - Hour of day
    #  - Is it targeted
    #  - Is there a previous message from this user?
    #  - Did the previous message from this user have a target?
    #  - How long ago was the previous message from this user in messages?
    #  - How long ago was the previous message from this user in minutes?
    #  - Are they a bot?
    #  - Is the message after from the same user?
    #  - Is the message before from the same user?
    # Both
    #  - Is this a self-link?
    #  - How far apart in messages are the two?
    #  - How far apart in time are the two?
    #  - Does the link target the query user?
    #  - Does the query target the link user?
    #  - none in between from src?
    #  - none in between from target?
    #  - previously src addressed target?
    #  - future src addressed target?
    #  - src addressed target in between?
    #  - previously target addressed src?
    #  - future target addressed src?
    #  - target addressed src in between?
    #  - are they the same speaker?
    #  - do they have the same target?
    #  - Do they have words in common?

####################
## data exploration

def arcs2dict(one_instance):
    """turns a list of pair of indices into an adjacency dictionary
    """
    gold = {}
    for (i,j) in one_instance["arcs"].indices:
        if i in gold:
            gold[i].append(j)
        else:
            gold[i] = [j]
    return gold

def display_instance(
    one_instance,
    start=0,
    nb = 200,
    turn_key = "lines",
    prev = False, # highlight that head is previous turn (or color?)
    ):
    
    gold = arcs2dict(one_instance)
    for i,turn in enumerate(one_instance[turn_key][start:start+nb]):
        k = i + start
        head = gold.get(k,"NONE")
        if head == [k]: head = "SELF-LOOP"
        if head==[k-1]: head = "PREV"
        print(k,"->",head,turn.tokens)
    
    
    

if __name__=="__main__":
    from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer, PretrainedTransformerTokenizer
    from allennlp.data.token_indexers import PretrainedTransformerIndexer
    from allennlp.data.vocabulary import Vocabulary
    #from allennlp.training.trainer import Trainer
    #from allennlp.data.iterators import BucketIterator
    from allennlp.common import Params

    token_indexers = {"tokens": SingleIdTokenIndexer()}
    tokenizer_cfg = Params({"word_splitter": {"language": "en"}})
    tokenizer = Tokenizer.from_params(tokenizer_cfg)
    reader = ChatReader(
        tokenizer=tokenizer,
        token_indexers=token_indexers,
        clip=50,
        raw=False#True
        )
    train_instances = reader.read("../data/train")
    dev_instances = reader.read("../data/dev")
    test_instances = reader.read("../data/test")
    #vocab = Vocabulary.from_instances([x["sentence"] for x in train_instances])
    #vocab = Vocabulary.from_instances(train_instances)
    #vocab = Vocabulary.from_instances(dev_instances)
    #vocab = Vocabulary.from_instances(test_instances)
    
    # preprocessing
    t1 = "<Sven_vB> TJ-, good news is I have text I/O now. bad news is I seem to not have that overlay kernel module. https://paste.ubuntu.com/p/q9zGJgJczk/"
    t2 = "<lupulo> frazr: this thread could be useful https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=788050"
    t3 = "=== vassie [~vassie@195.153.177.75]  has joined #ubuntu"
    print(preprocess_raw_turn(t1))
    print(preprocess_raw_turn(t2))
    print(preprocess_raw_turn(t3))
    
    
