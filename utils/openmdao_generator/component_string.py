from textwrap import indent
from . import derivative as d
import subprocess


def component_str(c_name, inputs, outputs, comp_f, black=True):
    """
    :param c_name: component name
    :param inputs: list of input variables "renamed"
    :param outputs: list of output variables "renamed"
    :param comp_f: edited computation function
    :param black: boolean than enables to use black to reformat the string
    :return: a string containing the code of an om.Component
    """
    s = ""
    s += "class " + c_name + "(om.ExplicitComponent):\n\n"
    s += "\tdef setup(self):\n"
    for i in range(0, len(inputs)):
        if inputs[i].unit == "None":
            s += "\t\tself.add_input('{}', val={})\n".format(
                inputs[i].name, inputs[i].val
            )
        else:
            s += "\t\tself.add_input('{}', val={}, units='{}')\n".format(
                inputs[i].name, inputs[i].val, inputs[i].unit
            )
    for i in range(0, len(outputs)):
        if outputs[i].unit == "None":
            s += "\t\tself.add_output('{}')\n".format(outputs[i].name)
        else:
            s += "\t\tself.add_output('{}', units='{}')\n".format(
                outputs[i].name, outputs[i].unit
            )
    s += "\n\tdef setup_partials(self):\n\t\tself.declare_partials('*', '*', method='fd')\n\n"
    s += "\tdef compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):\n"
    s += indent(comp_f, prefix="\t\t") + "\n"

    if black:
        try:
            byt = subprocess.check_output("black --code " + '"' + s + '"')
            byt = byt.replace(b"\r", b"")
            s = byt.decode("ascii")
        except subprocess.CalledProcessError:
            pass
    return s


def component_str_derivative(c_name, inputs, outputs, const, comp_f, pack, black=True):
    """
    :param c_name: component name
    :param inputs: list of input variables "renamed"
    :param outputs: list of output variables "renamed"
    :param const: list of constants present in the component
    :param comp_f: edited computation function
    :param pack: list of packages that the user wants to import (instances of the Pack class)
    :param black: boolean than enables to use black to reformat the string
    :return: a string containing the code of an om.Component
    """
    s = ""
    s += "class " + c_name + "(om.ExplicitComponent):\n\n"
    s += "\tdef setup(self):\n"
    for i in range(0, len(inputs)):
        if inputs[i].unit == "None":
            s += "\t\tself.add_input('{}', val={})\n".format(
                inputs[i].name, inputs[i].val
            )
        else:
            s += "\t\tself.add_input('{}', val={}, units='{}')\n".format(
                inputs[i].name, inputs[i].val, inputs[i].unit
            )
    for i in range(0, len(outputs)):
        if outputs[i].unit == "None":
            s += "\t\tself.add_output('{}')\n".format(outputs[i].name)
        else:
            s += "\t\tself.add_output('{}', units='{}')\n".format(
                outputs[i].name, outputs[i].unit
            )
    s += "\n\tdef setup_partials(self):\n"
    for var_out in outputs:
        for p in var_out.param:
            if p.deleted and not p.output:
                var_out.param.remove(p)
        input_param = d.get_input_param(var_out, [])
        if len(input_param) > 0:
            if len(input_param) == 1:
                param_name = "'{}'".format(input_param[0].name)
            else:
                param_name = "["
                for p in input_param:
                    param_name += "'{}', ".format(p.name)
                param_name = param_name[:-2]
                param_name += "]"
            s += "\t\tself.declare_partials('{}', {})\n".format(
                var_out.name, param_name
            )
    s += "\n\tdef compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):\n"
    s += indent(comp_f, prefix="\t\t") + "\n"
    s += "\tdef compute_partials(self, inputs, J, **kwargs):\n"
    for i in range(len(inputs)):
        s += "\t\t{} = inputs['{}']\n".format(inputs[i].symbol, inputs[i].name)
    s += "\n"
    for out in outputs:
        input_param = d.get_input_param(out, [])
        der = d.get_derivatives(out, pack, const)
        for j in range(len(input_param)):
            s += (
                "\t\tJ['{}', '{}'] = ".format(out.name, input_param[j].name)
                + der[j]
                + "\n"
            )
        s += "\n"

    if black:
        try:
            byt = subprocess.check_output("black --code " + '"' + s + '"')
            byt = byt.replace(b"\r", b"")
            s = byt.decode("ascii")
        except subprocess.CalledProcessError:
            pass
    return s
