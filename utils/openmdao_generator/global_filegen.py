from .variable_recognition import edit_function
from .component_string import *
from . import global_string_gen as gs
import os
from . import parse_pack as pp


def open_file(f_name):
    """
    :param f_name: name of the file
    :return: a new file with as many (1) as needed to not already exist
    """
    try:
        f = open("{}.py".format(f_name), "x")
        return f, f_name
    except IOError:
        return open_file(f_name + "(1)")


def generate_file(result, pack, d_check):
    """
    !!! still used !!!
    :param pack: a list of packages that the user wants to import (instances of the Pack class)
    :param result: a list of group names associated with their list_of_components containing CompData instances
    :param pack: list of packages that the user wants to import (instances of the Pack class)
    :param d_check: boolean to specify if derivatives are to be analytic
    :return: generates one file per group with python code
    """
    if result[0][0] is None and len(result) == 1:
        comp = result[0][1]
        f_name = comp[0].name
        f, f_name = open_file(f_name)
        f.write("import openmdao.api as om\n")
        if len(pack) > 0:
            f.write(pp.string_pack(pack))
        f.write("\n\n")
        for i in range(len(comp)):
            comp_f = comp[i].equation
            var_in, var_out, const = comp[i].var_in, comp[i].var_out, comp[i].constants
            comp_f = edit_function(var_in, var_out, comp_f)
            c_name = comp[i].name
            add_component(f, c_name, var_in, var_out, const, comp_f, pack, d_check)
        f.close()
        os.system("black " + f_name + ".py")
    else:
        for i in range(len(result)):
            f_name = result[i][0]
            f, f_name = open_file(f_name)
            f.write("import openmdao.api as om\n")
            f.write("import fastoad.api as oad\n")
            if len(pack) > 0:
                f.write(pp.string_pack(pack))
            f.write("\n\n")
            f.write("@oad.RegisterOpenMDAOSystem('drone')\n")  # register fast-oad model
            add_group(
                f, result[i][0], [[comp.name, comp.name] for comp in result[i][1]], 0
            )
            for comp_data in result[i][1]:
                comp_f = comp_data.equation
                var_in, var_out, const = (
                    comp_data.var_in,
                    comp_data.var_out,
                    comp_data.constants,
                )
                comp_f = edit_function(var_in, var_out, comp_f)
                c_name = comp_data.name
                add_component(f, c_name, var_in, var_out, const, comp_f, pack, d_check)
            f.close()
            os.system("black " + f_name + ".py")


def new_generate_file(hg_data, pack, d_check):
    """
    :param hg_data: input parsed data as Hg_data
    :param pack: a list of packages that the user wants to import (instances of the Pack class)
    :param d_check: boolean to specify if derivatives are to be analytic
    :return: generates as many files as there are highest level groups
    """
    if hg_data.last:
        generate_file(hg_data.children, pack, d_check)
    else:
        for child in hg_data.children:
            s = gs.rec_gen_string(child, pack, d_check, black=False)
            f_name = child.name
            f, f_name = open_file(f_name)
            f.write("import openmdao.api as om\n")
            if len(pack) > 0:
                f.write(pp.string_pack(pack))
            f.write("\n\n")
            f.write(s)
            f.close()
            os.system("black " + f_name + ".py")


def add_component(f, c_name, inputs, outputs, const, comp_f, pack, d_check):
    """
    :param f: target file
    :param c_name: component name
    :param inputs: list of input variables and their associated names
    :param outputs: list of output variables and their associated names
    :param const: list of constants in the component
    :param comp_f: edited computation function
    :param pack: list of packages that the user wants to import (instances of the Pack class)
    :param d_check: boolean to specify if derivatives are to be analytic
    :return: writes in the target file the code for the selected component
    """
    if d_check:
        f.write(
            component_str_derivative(c_name, inputs, outputs, const, comp_f, pack)
            + "\n"
        )
    else:
        f.write(component_str(c_name, inputs, outputs, comp_f) + "\n")


def add_group(f, g_name, subsystems, init):
    """
    :param f: target file
    :param g_name: group name
    :param subsystems: list of names of the components of the group
    :param init: initialization function if needed
    :return: writes in the target file the code for the selected group (without the code for the components)
    """

    f.write("class {}(om.Group):\n\n".format(g_name))
    if init != 0:
        f.write("\tdef initialize(self):\n")
        f.write(indent(init, prefix="\t\t"))
        f.write("\n\n")
    f.write("\tdef setup(self):\n")
    for i in range(len(subsystems)):
        f.write(
            '\t\tself.add_subsystem("{}", {}(), promotes=["*"])\n'.format(
                subsystems[i][0], subsystems[i][1]
            )
        )
    f.write("\n\n")
