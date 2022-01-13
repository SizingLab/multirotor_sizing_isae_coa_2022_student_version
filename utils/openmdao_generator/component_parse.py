import re


def is_empty(str):
    """
    :param str: input string
    :return: a boolean, True if the string is empty meaning contains only spaces, tabs and line breaks
    """
    for l in str:
        if l != "\s" and l != "\n" and l != "\t":
            return False
    return True


def format_comp(comp):
    """
    :param comp: component name
    :return: removes spaces before and after the name
    """
    while len(comp) > 1 and comp[0] == " ":
        comp = comp[1:]
    while len(comp) > 1 and comp[-1] == " ":
        comp = comp[:-1]
    return comp


def format_str(str):
    """
    :param str: input string
    :return: a copy of the input string but with no empty lines and no tabs
    """
    lines = str.split("\n")
    non_empty_lines = [line for line in lines if line.strip() != ""]
    str_ = ""
    for line in non_empty_lines:
        str_ += line + "\n"
    return str_


def parse_comp(str):
    """
    :param str: input string
    :return: a list of component names and their associated raw equations
    """
    p_str = re.split(r"#%\s", str)
    components = []

    for x in p_str:
        if is_empty(x):
            p_str.remove(x)

    for c in p_str:
        comp = c.split("\n", 1)
        comp[0] = format_comp(comp[0])
        comp[1] = format_str(comp[1])
        components.append([comp[0], comp[1]])

    return components


TEXT = (
    "\n"
    "#% Component1\n"
    "\n"
    "# comment here\n"
    "x = y*3 +2\n"
    "z = w**2 +a*4\n"
    "\n"
    "#% Component2\n"
    "a = b + c*2\n"
    "d = e + f\n"
    "\n"
)


def main():
    print(parse_comp(TEXT))


if __name__ == "__main__":
    main()
