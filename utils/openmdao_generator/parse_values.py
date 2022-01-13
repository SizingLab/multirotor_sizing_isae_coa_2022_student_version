def parse_values(cell, d_values):
    """
    :param cell: ipython cell content as a string
    :param d_values: default values for output parameters
    :return: gets the default values in the specified cell
    """
    lines = cell.split("\n")
    for line in lines:
        if "=" in line:
            val = line.split("=")
            if len(val) == 2:
                val[0] = val[0].strip()
                val[1] = val[1].strip()
                d_values[val[0]] = val[1]


def parse_imports(cell):
    """
    :param cell: ipython cell content as a string
    :return: removes #imports
    """
    cell_ = cell.split("\n", 1)[1]
    return cell_


def format_imports(imports):
    res = ""
    for i in imports:
        res += i + ", "
    if len(res) >= 2:
        res = res[0:-2]
    return res


TEXT = "# Exclude\n" "x = 0\n" "y1 = 2\n" "z3 = 1.1e-3   \n"
IMPORTS = (
    "import numpy as np\n"
    "import math\n\n"
    "import matplotlib as plt\n"
    "# random comment"
)


def main():
    d_values = {}
    parse_values(TEXT, d_values)
    print(d_values)
    print(format_imports(parse_imports(IMPORTS)))


if __name__ == "__main__":
    main()
