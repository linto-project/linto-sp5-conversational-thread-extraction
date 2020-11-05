# -*- coding: utf-8 -*-


class IrcChat:
    "IRC Chat class."

    def __init__(self, filename):
        "docstring"
        self.filename = filename
        self.raw = None
        with open(filename, "r") as chat:
            self.raw = chat.readlines()

    def clean_data(self):
        "Clean raw data, delete timestemp, === and log marker in log message."
        for line in self.raw:
            clean_line = line.split()
            if "===" in clean_line[0]:
                if "@" in clean_line[2]:
                    clean_line.pop(2)
            clean_line = clean_line[1:]
            yield " ".join(clean_line)


def main(filename):
    "main test"
    irc_chat = IrcChat(filename)
    for line in irc_chat.clean_data():
        print(line)

if __name__ == '__main__':
    main("/home/sdelecra/data/irc-disentanglement/data/train/2007-01-29.train-c.ascii.txt")
