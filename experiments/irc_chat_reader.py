import logging
import os
import string
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
    sub_sequence: int/None, generate short contexts as instance, trying to predict links to the last segment of the sequence
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
        loop: bool = False,
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
        self.loop = loop
        
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
            inst_loops = [0]*2200
            
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
            if self.loop:
                for s, t in arcs:
                    #print(s, t, len(inst_loops))
                    if s == t:
                        inst_loops[s - first_line] = 1
            #else:
            # save = []
            # save2 = []
            inst_arcs = []
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
            
            for one_instance in self.text_to_instance(inst_tokens, inst_arcs, inst_loops,
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
        inst_loops: Iterable,
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
            data["first_line"] = metadata["first_line"]
            data["token_line"] = metadata["first_line"] + i
            fields["lines"] = seq_field
            if self.loop:
                #print(start, i, len(inst_loops), len(inst_tokens))
                #if len(inst_loops[start:i+1]) < 2:
                #    print(inst_tokens[start:i+1], inst_loops[start:i+1], inst_loops[i], metadata["file_source"])
#                 if "2011-11" in metadata["file_source"]:
#                     print(inst_loops[0:250])
                label = inst_loops[i]
                fields["loops"] = LabelField(label, skip_indexing=True)
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
        inst_loops: Iterable,
        sub_sequence:int = None,
        metadata = {"tokens":[]}
    ) -> Instance:
        if sub_sequence: 
            return self.target_short_sequence_instance(inst_tokens,inst_arcs,inst_loops,context_length=sub_sequence,metadata=metadata)
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
    
    
# recycling Kummerfeld code from here, to test feature extraction

# see also get_features which puts everything together, including pairwise features

# beurk: lists words 100x more common than user names to avoid detecting them as users in turn initial position
# and makes a global variable
from reserved_words import reserved


def update_user(users, user):
    if user in reserved:
        return
    all_digit = True
    for char in user:
        if char not in string.digits:
            all_digit = False
    if all_digit:
        return
    users.add(user.lower())

# extract user id from a speech turn (ascii version) and update user set
def update_users(line, users):
    if len(line) < 2:
        return
    user = line[1]
    if user in ["Topic", "Signoff", "Signon", "Total", "#ubuntu"
            "Window", "Server:", "Screen:", "Geometry", "CO,",
            "Current", "Query", "Prompt:", "Second", "Split",
            "Logging", "Logfile", "Notification", "Hold", "Window",
            "Lastlog", "Notify", 'netjoined:']:
        # Ignore as these are channel commands
        pass
    else:
        if line[0].endswith("==="):
            parts = ' '.join(line).split("is now known as")
            if len(parts) == 2 and line[-1] == parts[-1].strip():
                user = line[-1]
        elif line[0][-1] == ']':
            if user[0] == '<':
                user = user[1:]
            if user[-1] == '>':
                user = user[:-1]

        user = user.lower()
        update_user(users, user)
        # This is for cases like a user named |blah| who is
        # refered to as simply blah
        core = [char for char in user]
        while len(core) > 0 and core[0] in string.punctuation:
            core.pop(0)
        while len(core) > 0 and core[-1] in string.punctuation:
            core.pop()
        core = ''.join(core)
        update_user(users, core)

# Names two letters or less that occur more than 500 times in the data
common_short_names = {"ng", "_2", "x_", "rq", "\\9", "ww", "nn", "bc", "te",
"io", "v7", "dm", "m0", "d1", "mr", "x3", "nm", "nu", "jc", "wy", "pa", "mn",
"a_", "xz", "qr", "s1", "jo", "sw", "em", "jn", "cj", "j_"}


# extract potential addresses from a turn (line), given a list of users
def get_targets(line, users):
    targets = set()
    for token in line[2:]:
        token = token.lower()
        user = None
        if token in users and len(token) > 2:
            user = token
        else:
            core = [char for char in token]
            while len(core) > 0 and core[-1] in string.punctuation:
                core.pop()
                nword = ''.join(core)
                if nword in users and (len(core) > 2 or nword in common_short_names):
                    user = nword
                    break
            if user is None:
                while len(core) > 0 and core[0] in string.punctuation:
                    core.pop(0)
                    nword = ''.join(core)
                    if nword in users and (len(core) > 2 or nword in common_short_names):
                        user = nword
                        break
        if user is not None:
            targets.add(user)
    return targets


# extract features or information at turn level 
def lines_to_info(text_ascii,users=None):
    if users is None:
        users = set()
        for line in text_ascii:
            update_users(line, users)

    chour = 12
    cmin = 0
    info = []
    target_info = {}
    nexts = {}
    for line_no, line in enumerate(text_ascii):
        if line[0].startswith("["):
            user = line[1][1:-1]
            nexts.setdefault(user, []).append(line_no)

    prev = {}
    for line_no, line in enumerate(text_ascii):
        user = line[1]
        system = True
        if line[0].startswith("["):
            chour = int(line[0][1:3])
            cmin = int(line[0][4:6])
            user = user[1:-1]
            system = False
        is_bot = (user == 'ubottu' or user == 'ubotu')
        targets = get_targets(line, users)
        for target in targets:
            target_info.setdefault((user, target), []).append(line_no)
        last_from_user = prev.get(user, None)
        if not system:
            prev[user] = line_no
        next_from_user = None
        if user in nexts:
            while len(nexts[user]) > 0 and nexts[user][0] <= line_no:
                nexts[user].pop(0)
            if len(nexts[user]) > 0:
                next_from_user = nexts[user][0]

        info.append((user, targets, chour, cmin, system, is_bot, last_from_user, line, next_from_user))

    return info, target_info

def read_data(filenames, is_test=False):
    instances = []
    done = set()
    for filename in filenames:
        name = filename
        for ending in [".annotation.txt", ".ascii.txt", ".raw.txt", ".tok.txt"]:
            if filename.endswith(ending):
                name = filename[:-len(ending)]
        if name in done:
            continue
        done.add(name)
        text_ascii = [l.strip().split() for l in open(name +".ascii.txt")]
        text_tok = []
        for l in open(name +".tok.txt"):
            l = l.strip().split()
            if len(l) > 0 and l[-1] == "</s>":
                l = l[:-1]
            if len(l) == 0 or l[0] != '<s>':
                l.insert(0, "<s>")
            text_tok.append(l)
        info, target_info = lines_to_info(text_ascii)

        links = {}
        if is_test:
            for i in range(args.test_start, min(args.test_end, len(text_ascii))):
                links[i] = []
        else:
            for line in open(name +".annotation.txt"):
                nums = [int(v) for v in line.strip().split() if v != '-']
                links.setdefault(max(nums), []).append(min(nums))
        for link, nums in links.items():
            instances.append((name +".annotation.txt", link, nums, text_ascii, text_tok, info, target_info))
    return instances  
   
    
def get_time_diff(info, a, b):
    if a is None or b is None:
        return -1
    if a > b:
        t = a
        a = b
        b = t
    ahour = info[a][2]
    amin = info[a][3]
    bhour = info[b][2]
    bmin = info[b][3]
    if ahour == bhour:
        return bmin - amin
    if bhour < ahour:
        bhour += 24
    return (60 - amin) + bmin + 60*(bhour - ahour - 1)

cache = {}
def get_features(name, query_no, link_no, text_ascii, text_tok, info, target_info, do_cache):
    global cache
    if (name, query_no, link_no) in cache:
        return cache[name, query_no, link_no]

    features = []

    quser, qtargets, qhour, qmin, qsystem, qis_bot, qlast_from_user, qline, qnext_from_user = info[query_no]
    luser, ltargets, lhour, lmin, lsystem, lis_bot, llast_from_user, lline, lnext_from_user = info[link_no]

    # General information about this sample of data
    # Year
    for i in range(2004, 2018):
        features.append(str(i) in name)
    # Number of messages per minute
    start = None
    end = None
    for i in range(len(text_ascii)):
        if start is None and text_ascii[i][0].startswith("["):
            start = i
        if end is None and i > 0 and text_ascii[-i][0].startswith("["):
            end = len(text_ascii) - i - 1
        if start is not None and end is not None:
            break
    diff = get_time_diff(info, start, end)
    msg_per_min = len(text_ascii) / max(1, diff)
    cutoffs = [-1, 1, 3, 10, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= msg_per_min < end)

    # Query
    #  - Normal message or system message
    features.append(qsystem)
    #  - Hour of day
    features.append(qhour / 24)
    #  - Is it targeted
    features.append(len(qtargets) > 0)
    #  - Is there a previous message from this user?
    features.append(qlast_from_user is not None)
    #  - Did the previous message from this user have a target?
    if qlast_from_user is None:
        features.append(False)
    else:
        features.append(len(info[qlast_from_user][1]) > 0)
    #  - How long ago was the previous message from this user in messages?
    dist = -1 if qlast_from_user is None else query_no - qlast_from_user
    cutoffs = [-1, 0, 1, 5, 20, 1000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= dist < end)
    #  - How long ago was the previous message from this user in minutes?
    time = get_time_diff(info, query_no, qlast_from_user)
    cutoffs = [-1, 0, 2, 10, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= time < end)
    #  - Are they a bot?
    features.append(qis_bot)

    # Link
    #  - Normal message or system message
    features.append(lsystem)
    #  - Hour of day
    features.append(lhour / 24)
    #  - Is it targeted
    features.append(link_no != query_no and len(ltargets) > 0)
    #  - Is there a previous message from this user?
    features.append(link_no != query_no and llast_from_user is not None)
    #  - Did the previous message from this user have a target?
    if link_no == query_no or llast_from_user is None:
        features.append(False)
    else:
        features.append(len(info[llast_from_user][1]) > 0)
    #  - How long ago was the previous message from this user in messages?
    dist = -1 if llast_from_user is None else link_no - llast_from_user
    cutoffs = [-1, 0, 1, 5, 20, 1000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(link_no != query_no and start <= dist < end)
    #  - How long ago was the previous message from this user in minutes?
    time = get_time_diff(info, link_no, llast_from_user)
    cutoffs = [-1, 0, 2, 10, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= time < end)
    #  - Are they a bot?
    features.append(lis_bot)
    #  - Is the message after from the same user?
    features.append(link_no != query_no and link_no + 1 < len(info) and luser == info[link_no + 1][0])
    #  - Is the message before from the same user?
    features.append(link_no != query_no and link_no - 1 > 0 and luser == info[link_no - 1][0])

    # Both
    #  - Is this a self-link?
    features.append(link_no == query_no)
    #  - How far apart in messages are the two?
    dist = query_no - link_no
    features.append(min(100, dist) / 100)
    features.append(dist > 1)
    #  - How far apart in time are the two?
    time = get_time_diff(info, link_no, query_no)
    features.append(min(100, time) / 100)
    cutoffs = [-1, 0, 1, 5, 60, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= time < end)
    #  - Does the link target the query user?
    features.append(quser.lower() in ltargets)
    #  - Does the query target the link user?
    features.append(luser.lower() in qtargets)
    #  - none in between from src?
    features.append(link_no != query_no and (qlast_from_user is None or qlast_from_user < link_no))
    #  - none in between from target?
    features.append(link_no != query_no and (lnext_from_user is None or lnext_from_user > query_no))
    #  - previously src addressed target?
    #  - future src addressed target?
    #  - src addressed target in between?
    if link_no != query_no and (quser, luser) in target_info:
        features.append(min(target_info[quser, luser]) < link_no)
        features.append(max(target_info[quser, luser]) > query_no)
        between = False
        for num in target_info[quser, luser]:
            if query_no > num > link_no:
                between = True
        features.append(between)
    else:
        features.append(False)
        features.append(False)
        features.append(False)
    #  - previously target addressed src?
    #  - future target addressed src?
    #  - target addressed src in between?
    if link_no != query_no and (luser, quser) in target_info:
        features.append(min(target_info[luser, quser]) < link_no)
        features.append(max(target_info[luser, quser]) > query_no)
        between = False
        for num in target_info[luser, quser]:
            if query_no > num > link_no:
                between = True
        features.append(between)
    else:
        features.append(False)
        features.append(False)
        features.append(False)
    #  - are they the same speaker?
    features.append(luser == quser)
    #  - do they have the same target?
    features.append(link_no != query_no and len(ltargets.intersection(qtargets)) > 0)
    #  - Do they have words in common?
    ltokens = set(text_ascii[link_no])
    qtokens = set(text_ascii[query_no])
    common = len(ltokens.intersection(qtokens))
    if link_no != query_no and len(ltokens) > 0 and len(qtokens) > 0:
        features.append(common / len(ltokens))
        features.append(common / len(qtokens))
    else:
        features.append(False)
        features.append(False)
    features.append(link_no != query_no and common == 0)
    features.append(link_no != query_no and common == 1)
    features.append(link_no != query_no and common > 1)
    features.append(link_no != query_no and common > 5)
    
    # Convert to 0/1
    final_features = []
    for feature in features:
        if feature == True:
            final_features.append(1.0)
        elif feature == False:
            final_features.append(0.0)
        else:
            final_features.append(feature)

    if do_cache:
        cache[name, query_no, link_no] = final_features
    return final_features
###################################
# end of Kummerfeld code recycling
###################################
    
    
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
        raw=True,
        sub_sequence=1,
        loop=True,
        )
    #train_instances = reader.read("../data/train")
    dev_instances = reader.read("../data/dev")
    j = 5
    for i in range(len(dev_instances)):
        if "2011-11-13_02" in dev_instances[i]["metadata"].metadata["file_source"]:
            print(i)
            print(dev_instances[i]["metadata"].metadata)
            print(dev_instances[i]["lines"][-1])
            print("label = ", dev_instances[i]["loops"].label)
            j -= 1
        if j < 0:
            break
    #test_instances = reader.read("../data/test")
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
    
    
