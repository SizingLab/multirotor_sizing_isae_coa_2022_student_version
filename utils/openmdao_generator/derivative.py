from sympy import *
from .variable_recognition import Variable, Constant
from .parse_pack import Pack
import re

LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def get_derivatives(var, pack, const):
    """
    :param var: output variable to get the derivatives from
    :param pack: list of packages imported
    :param const: list of the constants to handle
    :return: a list of expressions containing derivatives relative to every input variable that affects var
    """
    der = []
    ns = {}
    var.equation = format_equation(var.equation, pack)
    for p in get_input_param(var, []):
        ns[p.symbol] = Symbol(p.symbol)
    ns_0 = ns
    for c in const:
        ns_0[c.short] = Symbol(c.short)
    eq_str = parse_eq_rec(var, pack)
    try:
        eq = sympify(eq_str, locals=ns_0)
        for key in ns:
            derv = str(diff(eq, ns[key]))
            der.append(format_derivative(derv, const))
    except SympifyError:
        for key in ns:
            der.append("# sympy could not parse the equation")
    return der


def parse_eq_rec(var, pack):
    """
    :param var: output variable to get the derivatives from
    :param pack: list of packages imported
    :return: an equation where intermediate variables are recursively replaced by their expression relative to input
    variables
    """
    eq_str = format_equation(var.equation, pack)
    for p in var.param:
        if p.output:
            repl = "(" + parse_eq_rec(p, pack) + ")"
            eq_str = re.sub(r"(?<=\b)" + p.symbol + r"(?=\b)", repl, eq_str)
    return eq_str


def get_input_param(var, param):
    """
    :param var: output variable to get the derivatives from
    :param param: lists of variables var depends on
    :return: a list of input variables that var really depends on (without intermediate variables)
    """
    param_ = param
    for p in var.param:
        if p not in param_:
            if not p.output:
                param_.append(p)
            else:
                param_ = get_input_param(p, param_)
    return param_


def format_equation(eq_str, pack):
    """
    :param eq_str: the expression to format
    :param pack: list of packages imported
    :return: an expression without the package prefixes so that sympy can handle it
    """
    for p in pack:
        if p.short:
            pre = p.nick + "."
        else:
            pre = p.name + "."
        eq_str = eq_str.replace(pre, "")
    return eq_str


def format_derivative(der, const):
    """
    :param der: a list of expressions containing derivatives relative to every input variable that affects var
    :param const: list of the constants to handle
    :return: adds back the package prefixes to the functions and constants
    (the function will always get np. but the constants keep the original package prefix)
    """
    der_list = []
    index = 0
    for i in range(1, len(der)):
        if der[i - index] == "(" and i > index:
            if der[i - index - 1] in LETTERS:
                func = der[i - index - 1]
                j = 2
                while j <= i - index and der[i - index - j] in LETTERS:
                    func = der[i - index - j] + func
                    j += 1
                der_list.append([der[: i - index], func])
                der = der[i - index :]
                index = i + 1
    der_fin = ""
    for elt in der_list:
        st = elt[0]
        func = elt[1]
        new_func = "np." + func
        st = st[: -len(func)]
        st += new_func
        der_fin += st
    der_fin += der

    for c in const:
        cs = c.short
        if cs in der_fin:
            der_c = der_fin
            l = len(cs)
            der_c_list = []
            index = 0
            for i in range(len(der_c)):
                j = 0
                while j <= i - index and der_c[i - index - j] == c.short[-j - 1]:
                    if j == l - 1:
                        beg = False
                        if i - index - j == 0:
                            beg = True
                        elif der_c[i - index - j - 1] not in LETTERS:
                            beg = True
                        end = False
                        if i - index == len(der_c) - 1:
                            end = True
                        elif der_c[i - index + 1] not in LETTERS:
                            end = True
                        if beg and end:
                            der_c_list.append(der_c[: i - index + 1])
                            der_c = der_c[i - index + 1 :]
                            index = i + 1
                        break
                    j += 1
            der_fin = ""
            for st in der_c_list:
                st = st[:-l]
                st += c.full
                der_fin += st
            der_fin += der_c

    return der_fin


DER = "pi*pisinpicostanpi + pi /(x+(tan(y)*cos(x)) + pi"
CONST = [Constant("np.pi")]

p1 = Pack(name="numpy", short=True, nick="np")
p2 = Pack(name="mat", short=False, nick="")
packs = [p1, p2]

expr = "np.log(y) + mat.exp(z**2) + np.tan(u)"
expr2 = "y**2 + np.cos(z)"

var1 = Variable("x", output=True)
var2 = Variable("y")
var3 = Variable("z")
var4 = Variable("u", output=True)

var1.add_param(var2)
var1.add_param(var3)
var1.add_param(var4)
var4.add_param(var2)
var4.add_param(var3)

var1.equation = expr
var4.equation = expr2


def main():
    print(get_derivatives(var1, packs, []))
    print(format_derivative(DER, CONST))
    # print(format_equation("np.cos(np.euler**x)", packs))


if __name__ == "__main__":
    main()
