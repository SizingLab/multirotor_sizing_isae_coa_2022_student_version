class Pack:
    def __init__(self, name="default_pack", short=False, nick="def_pack"):
        self.name = name
        self.short = short
        self.nick = nick
        self.elements = []

    def add_element(self, elt):
        self.elements.append(elt)

    def __str__(self):
        return (
            "|"
            + self.name
            + "||"
            + str(self.short)
            + "||"
            + self.nick
            + "||"
            + self.elements
            + "|"
        )


def parse_pack(str):
    """
    :param str: string to get packages from
    :return: a list of Pack class elements
    """
    spt = str.split("\n")
    pack = []
    for x in spt:
        x = x.strip()
        if "import" in x:
            words = x.split(" ")
            if "" in words:
                words.remove("")
            if "," in words:
                words.remove(",")
            if words[0] == "import" and len(words) == 2:
                name = words[1]
                short = False
                nick = ""
                pack.append(Pack(name, short, nick))
            elif len(words) == 4 and words[0] == "import" and words[2] == "as":
                name = words[1]
                short = True
                nick = words[3]
                pack.append(Pack(name, short, nick))
            elif len(words) >= 4 and words[0] == "from" and words[2] == "import":
                name = words[1]
                short = False
                nick = ""
                words_ = words[3:]
                p = Pack(name, short, nick)
                for elt in words_:
                    elt = elt.replace(",", " ")
                    elt = elt.strip()
                    p.add_element(elt)
                pack.append(p)
    return pack


def string_pack(pack):
    s = ""
    for p in pack:
        if p.short:
            s += "import {} as {}\n".format(p.name, p.nick)
        elif p.elements:
            elt = ""
            for e in p.elements:
                elt += e + ", "
            elt = elt[:-2]
            s += "from {} import {}\n".format(p.name, elt)
        else:
            s += "import {}\n".format(p.name)
    return s


PACKS = "import numpy as np \nfrom math import sin, cos  , tan \n import importlib\nimport ipyvuetify"


def main():
    print(string_pack(parse_pack(PACKS)))


if __name__ == "__main__":
    main()
