# -*- coding: utf-8 -*-
# Author: Sebastien Delecraz

import os
import random
import csv
import sys
from typing import List, Set, Tuple
from tqdm import tqdm
from irc_chat import IrcChat

random.seed(1234)


def get_links(filename: str, test=False) -> List[Tuple[int, int]]:
    links = set()
    with open(filename, 'r') as file_stream:
        for line in file_stream:
            source, target, _ = line.split()
            if source != target:
                if test and int(source) < 1000 and int(target) < 1000:
                    continue
                links.add(f"{source}-{target}")
    return links


def generate_labeled_pairs(
        conversations: List[str],
        links: Set[Tuple[int, int]],
        label: int,
        test=False) -> List[Tuple[str, str, int]]:
    pairs = list()
    ll = list(links)
    ll.sort()
    for link in ll:
        if test:
            if source < 1000 or target < 1000:
                continue
        source, target = link.split("-")
        context = conversations[int(source)]
        utterance = conversations[int(target)]
        pairs.append([source, target, context, utterance, label])
    return pairs


def generate_5_negative_links(links: Set[Tuple[int, int]],
                            conv_number: int) -> Set[Tuple[int, int]]:
    neg_links = set()
    for link in links:
        source, target = map(int, link.split("-"))
        for new_target in range(max(source, target - 2),
                                min(target + 3, conv_number)):
            neg_link = f"{source}-{new_target}"
            if neg_link not in links and source != new_target:
                neg_links.add(neg_link)
    return neg_links

def generate_10_negative_links(links: Set[Tuple[int, int]],
                            conv_number: int) -> Set[Tuple[int, int]]:
    neg_links = set()
    for link in links:
        source, target = map(int, link.split("-"))
        for new_target in range(max(source, target - 5),
                                min(target + 6, conv_number)):
            neg_link = f"{source}-{new_target}"
            if neg_link not in links and source != new_target:
                neg_links.add(neg_link)
    return neg_links

def generate_eq_negative_links(links: Set[Tuple[int, int]],
                            conv_number: int) -> Set[Tuple[int, int]]:
    neg_links = set()
    for link in links:
        source, target = map(int, link.split("-"))
        rand_count = 0
        while "negative link not found":
            t = random.randint(-5, 4)
            new_target = min(max(source+1, target + t), conv_number - 1)
            neg_link = f"{source}-{new_target}"
            rand_count += 1
            if neg_link not in links and rand_count < 10 and source != new_target:
                neg_links.add(neg_link)
                break
            if rand_count >= 10:
                break
    return neg_links


# def conversations_as_list(filename: str) -> List[str]:
#     conversations = list()
#     with open(filename, 'r') as conv_stream:
#         for line in conv_stream:
#             conversations.append(line.strip("</s>\n").strip())
#     return conversations




def generate_full_negative_links(links: Set[str],
                                 conv_number: int) -> Set[str]:
    neg_links = set()
    for i in range(1000, conv_number):
        for j in range(i + 1, conv_number):
            neg_link = f"{i}-{j}"
            if neg_link in links:
                continue
            else:
                neg_links.add(neg_link)
    return neg_links


negative_links_generator = {
    "eq": generate_eq_negative_links,
    "5": generate_5_negative_links,
    "10": generate_10_negative_links,
    "full" : generate_full_negative_links,
}

def generate_local_pairs(filepath: str, extension:str, negative_num: str, test=False) -> List[Tuple[str, str, int]]:
    filename = ".".join([filepath, extension])
    conversations = list(IrcChat(filename).clean_data())
    links_filename = filename.replace(extension.split(".")[0], "annotation")
    links = get_links(links_filename, test)
    neg_links = negative_links_generator[negative_num](links, len(conversations))
    for l in links:
        s, t = l.split("-")
        if s == t:
            print(filename, s, t)
            sys.exit(1)
    for l in neg_links:
        s, t = l.split("-")
        if s == t:
            print(filename, s, t)
            sys.exit(1)
    pairs = generate_labeled_pairs(conversations, links, 1)
    pairs.extend(generate_labeled_pairs(conversations, neg_links, 0))
    return pairs


def get_files_id(data_folder: str):
    folder_files = os.listdir(data_folder)
    files_id = set()
    for filename in folder_files:
        *file_id, ext, txt = filename.split(".")
        files_id.add(".".join(file_id))
    ll = list(files_id)
    ll.sort()
    return ll


def generate_corpus(data_folder: str,
                    file_extension: str,
                    output_file: str,
                    corpus_set: str,
                    negative_num: str) -> None:
    pairs = list()
    for file_id in tqdm(get_files_id(data_folder), desc=f"Generate corpus {corpus_set}"):
        if "test" in corpus_set:
            local_pairs = generate_local_pairs(os.path.join(data_folder, file_id), file_extension,  negative_num, test=True)
        local_pairs = generate_local_pairs(os.path.join(data_folder, file_id), file_extension,  negative_num)
        pairs.extend([[file_id] + p for p in local_pairs])
    random.seed(1234)
    random.shuffle(pairs)
    data = [[k, *ex] for (k, ex) in enumerate(pairs)]
    data.insert(0, ["index", "file_id", "source", "target", "sentence1", "sentence2", "label"])
    with open(output_file, "w") as out_stream:
        tsv_writer = csv.writer(out_stream, delimiter='\t')
        tsv_writer.writerows(data)

# def generate_dev_corpus(data_folder: str,
#                     file_extension: str,
#                     output_file: str,
#                     corpus_set: str,
#                     negative_num: str) -> None:
#     pairs = list()
#     for file_id in tqdm(get_files_id(data_folder), desc=f"Generate corpus {corpus_set}"):
#         filename = ".".join([filepath, extension])
#         conversations = list(IrcChat(filename).clean_data())
#         links_filename = filename.replace(extension.split(".")[0], "annotation")
#         links = get_links(links_filename)
#         local_pairs = generate_local_pairs(os.path.join(data_folder, file_id), file_extension,  negative_num)
#         pairs.extend([[file_id] + p for p in local_pairs])
#     random.seed(1234)
#     random.shuffle(pairs)
#     data = [[k, *ex] for (k, ex) in enumerate(pairs)]
#     data.insert(0, ["index", "file_id", "source", "target", "sentence1", "sentence2", "label"])
#     with open(output_file, "w") as out_stream:
#         tsv_writer = csv.writer(out_stream, delimiter='\t')
#         tsv_writer.writerows(data)


if __name__ == '__main__':
    """
    Usage: python generate_irc_corpus.py DATA_FOLDER OUTPUT_FOLDER FILE_EXTENSION NEG_NUM CORPUS_SET
    Example: python generate_irc_corpus.py irc-data-folder output-folder .tok.txt eq train
    """
    DATA_FOLDER = sys.argv[1]
    OUT_FOLDER = sys.argv[2]
    EXT = sys.argv[3]
    negative_num = sys.argv[4]
    corpus_set = sys.argv[5]
    # for corpus_set in ["train", "dev", "test"]:
    generate_corpus(os.path.join(DATA_FOLDER, corpus_set),
                    EXT,
                    os.path.join(DATA_FOLDER, OUT_FOLDER, corpus_set) + ".tsv",
                    corpus_set,
                    negative_num)
