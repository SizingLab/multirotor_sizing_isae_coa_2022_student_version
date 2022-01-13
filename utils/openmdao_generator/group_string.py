from textwrap import indent
from . import parse_pack as pp


def group_str(g_name, subsystems, init, pack, imports=False):
    """
    :param imports: boolean to specify if imports are required
    :param pack: list of packages that the user wants to imports (instances of the Pack class)
    :param g_name: group name
    :param subsystems: list of the components that are to be added to the group
    (component name, class name, package name)
    :param init: content of the initialization function, 0 if no function is required
    :return: a string containing the code of an om.Group
    """
    s = ""
    if imports:
        s += "import openmdao.api as om\n"
        if len(pack) > 0:
            s += pp.string_pack(pack)
        s += "\n\n"
    s += "class {}(om.Group):\n\n".format(g_name)
    if init != 0:
        s += "\tdef initialize(self):\n\n"
        s += indent(init, prefix="\t\t")
        s += "\n"
    s += "\tdef setup(self):\n"
    for i in range(len(subsystems)):
        s += '\t\tself.add_subsystem("{}", {}(), promotes=["*"])\n'.format(
            subsystems[i][0], subsystems[i][1]
        )
    return s
