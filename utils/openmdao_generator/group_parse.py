import re


def is_empty(str):
    """
    :param str: input string
    :return: a boolean, True if the string is empty meaning contains only spaces, tabs and line breaks
    """
    for l in str:
        if l != " " and l != "\n" and l != "\t":
            return False
    return True


def format_str(str):
    """
    :param str: input string
    :return: a copy of the input string but with no empty lines
    """
    lines = str.split("\n")
    non_empty_lines = [line for line in lines if line.strip() != ""]
    str_ = ""
    for line in non_empty_lines:
        str_ += line + "\n"
    return str_


def format_group(gr):
    """
    :param gr: group name
    :return: removes spaces before and after the name
    """
    while len(gr) > 1 and gr[0] == " ":
        gr = gr[1:]
    while len(gr) > 1 and gr[-1] == " ":
        gr = gr[:-1]
    return gr


def parse_group(str):
    """
    :param str: input string
    :return: a list of group names and their associated raw text
    """
    p_str = re.split(r"#%%\s", str)
    groups = []

    for x in p_str:
        if is_empty(x):
            p_str.remove(x)

    for c in p_str:
        g = c.split("\n", 1)
        g[0] = format_group(g[0])
        g[1] = format_str(g[1])
        groups.append([g[0], g[1]])

    return groups


TEXT = (
    "\n"
    "#%% Group1\n"
    "#% Component1\n"
    "\n"
    "\n"
    "x = y*3 +2\n"
    "z = w**2 +a*4\n"
    "\n"
    "#% Component2\n"
    "a = b + c*2\n"
    "d = e + f\n"
    "#%% Group2\n"
    "\n"
    "\n"
    "#% Component3\n"
    "\n"
    "\n"
    "x = y*3 +5\n"
    "z = w**2 +a*3\n"
    "\n"
    "#% Component4\n"
    "a = b + c*6\n"
    "d = e + f*3\n"
    "\n"
)


def main():
    print(parse_group(TEXT))


if __name__ == "__main__":
    main()
