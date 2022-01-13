from .global_string_gen import HGdata
from os import path

LS = []
HG_DATA = HGdata()


def set_globals(ls, hg_data):
    global LS
    LS = ls
    global HG_DATA
    HG_DATA = hg_data


def file_check(name):
    f_name = name + ".html"
    if path.isfile(f_name):
        return file_check(name + "(1)")
    else:
        return name
