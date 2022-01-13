import ipywidgets as widgets
import ipyvuetify as v
import ipysheet
from . import global_string_gen as gen_str
from .global_filegen import new_generate_file
from .parse_values import parse_values, parse_imports
from .delete import delete_var
from . import parse_pack as pp
from IPython.display import display, Markdown
import importlib
from . import n2_execute
from . import n2_globals

HG_DATA = []
RESULT = []
IN = []
D_VALUES = {}
DELETED_VAR = []
GEN_OR_PRINT = False
PACK = []
S = ""
LS = []

copy_but = v.Btn()

copy_field_1 = widgets.Text()

copy_field_2 = widgets.Text()

analyse_but = v.Btn()

derivative_check = v.Checkbox()

print_but = v.Btn()

gen_but = v.Btn()

n2_but = v.Btn()

pack_area = v.Textarea()

function = v.Textarea()

hb = widgets.Box()

vb = widgets.Box()


def init(in_):
    # Function to initialize the GUI
    global RESULT
    RESULT = []

    global IN
    IN = in_

    global D_VALUES
    D_VALUES = {}

    global copy_but
    copy_but = v.Btn(
        children=["Copy cells"], color="blue lighten-1", height="35px", width="250px"
    )
    copy_but.on_event("click", copy_click)

    global copy_field_1
    copy_field_1 = widgets.IntText(value="1", description="First cell", min=1, step=1)
    copy_field_1.layout.width = "140px"

    global copy_field_2
    copy_field_2 = widgets.IntText(value="2", description="Last cell", min=1, step=1)
    copy_field_2.layout.width = "140px"

    global analyse_but
    analyse_but = v.Btn(children=["Analyse"], color="blue lighten-1")
    analyse_but.on_event("click", analyse_click)

    global derivative_check
    derivative_check = v.Checkbox(label="Analytic derivatives", v_model=False)

    global print_but
    print_but = v.Btn(children=["Print code"], color="orange lighten-2")
    print_but.on_event("click", print_click)

    global gen_but
    gen_but = v.Btn(children=["Generate file"], color="green lighten-2")
    gen_but.on_event("click", gen_click)

    global n2_but
    n2_but = v.Btn(children=["Generate n2 diagram"], color="purple lighten-2")
    n2_but.on_event("click", n2_click)
    n2_but.disabled = True

    global pack_area
    pack_area = v.Textarea(
        v_model="import numpy as np",
        label="Packages to import",
        placeholder="import pack1 as pk1\nimport pack2",
        clearable=True,
        rounded=True,
        auto_grow=True,
        row=15,
        background_color="blue lighten-4",
    )

    global function
    function = v.Textarea(
        v_model="#% Component1\ny = x + 1",
        placeholder="Your equations here",
        label="Equations",
        clearable=True,
        rounded=True,
        auto_grow=False,
        rows=15,
        background_color="blue lighten-4",
    )

    global hb
    hb = widgets.Box(children=[copy_but, copy_field_1, copy_field_2])
    hb.layout.display = "flex"
    hb.layout.margin = "30px 10px 10px 50px"
    hb.layout.border_style = "solid"
    hb.layout.border_width = "5px"
    hb.layout.justify_content = "center"
    hb.layout.align_items = "center"
    hb.layout.align_content = "center"
    hb.layout.width = "90%"

    global vb
    vb = widgets.Box(children=[hb, function, pack_area, analyse_but])
    vb.layout.display = "flex"
    vb.layout.flex_flow = "column"
    vb.layout.align_items = "stretch"
    vb.layout.border_style = "solid"
    vb.layout.border_width = "5px"
    vb.layout.border_color = "black"
    return vb


def copy_click(widget, event, data):
    # Function called when the "copy cells" button is pressed
    n1 = int(copy_field_1.value)
    n2 = int(copy_field_2.value) + 1
    copy = ""
    global D_VALUES
    imports = ""
    if 0 < n1 < n2 <= len(IN):
        for cell in IN[n1:n2]:
            if len(cell) >= 6:
                if cell[0:6] != "# Init":
                    if len(cell) >= 8:
                        if cell[0:8] != "# Import":
                            if len(cell) >= 9:
                                if cell[0:9] != "# Exclude":
                                    copy += cell + "\n\n"
                        else:
                            imports += parse_imports(cell)
                else:
                    parse_values(cell, D_VALUES)
            else:
                copy += cell + "\n\n"
    function.v_model = copy
    if len(imports) > 0:
        pack_area.v_model = imports


def inner_analysis_in(c, var_in, i, group_cells, del_but):
    group_cells.append([var_in.symbol, []])

    del_but += [v.Btn(children=["Delete"], color="blue lighten-2")]

    def del_click(widget, event, data):
        global DELETED_VAR
        if widget.children[0] == "Delete":
            DELETED_VAR.append(i)
            widget.children = ["Deleted"]
            widget.color = "red lighten-2"
        else:
            DELETED_VAR.remove(i)
            widget.children = ["Delete"]
            widget.color = "blue lighten-2"

    del_but[i].on_event("click", del_click)

    ipysheet.cell(i, 0, c.group, background_color="#EEEEEE", read_only=True)
    ipysheet.cell(i, 1, c.name, background_color="#EEEEEE", read_only=True)
    group_cells[-1][1] += [ipysheet.cell(i, 2, var_in.symbol)]
    group_cells[-1][1] += [ipysheet.cell(i, 3, var_in.name)]
    ipysheet.cell(i, 4, "input", background_color="#8EFF9B", read_only=True)
    group_cells[-1][1] += [ipysheet.cell(i, 5, var_in.unit)]
    group_cells[-1][1] += [ipysheet.cell(i, 6, var_in.val)]
    ipysheet.cell(i, 7, del_but[i])
    if var_in.symbol in D_VALUES:
        ipysheet.cell(i, 6, D_VALUES[var_in.symbol])


def inner_analysis_out(c, var_out, i, group_cells, del_but):
    group_cells.append([var_out.symbol, []])

    del_but += [v.Btn(children=["Delete"], color="blue lighten-2")]

    def del_click(widget, event, data):
        global DELETED_VAR
        if widget.children[0] == "Delete":
            DELETED_VAR.append(i)
            widget.children = ["Deleted"]
            widget.color = "red lighten-2"
        else:
            DELETED_VAR.remove(i)
            widget.children = ["Delete"]
            widget.color = "blue lighten-2"

    del_but[i].on_event("click", del_click)

    ipysheet.cell(i, 0, c.group, background_color="#EEEEEE", read_only=True)
    ipysheet.cell(i, 1, c.name, background_color="#EEEEEE", read_only=True)
    group_cells[-1][1] += [ipysheet.cell(i, 2, var_out.symbol)]
    group_cells[-1][1] += [ipysheet.cell(i, 3, var_out.name)]
    ipysheet.cell(i, 4, "output", background_color="#FFB48E", read_only=True)
    group_cells[-1][1] += [ipysheet.cell(i, 5, var_out.unit)]
    group_cells[-1][1] += [
        ipysheet.cell(i, 6, "", background_color="#EEEEEE", read_only=True)
    ]
    ipysheet.cell(i, 7, del_but[i])


def analyse_click(widget, event, data):
    # Function called when the "analyse" button is pressed
    analyse_but.loading = True
    analyse_but.disabled = True

    global PACK
    PACK = pp.parse_pack(pack_area.v_model)

    in_str = function.v_model
    hg_data = gen_str.recursive_parse(in_str, PACK)
    result = gen_str.aggregate_result(hg_data)
    n = 0
    for g in result:
        for c in g[1]:
            n += len(c.var_in)
            n += len(c.var_out)

    headers = [
        "Group Name",
        "Component Name",
        "Variable Detected",
        "Variable Name",
        "Input/Output",
        "Units",
        "Default Value",
        "Delete Variable",
    ]

    sheet = ipysheet.sheet(columns=8, rows=n, row_headers=False, column_headers=headers)

    i = 0
    del_but = []
    for g in result:
        group_cells = []
        for c in g[1]:
            for var_in in c.var_in:
                inner_analysis_in(c, var_in, i, group_cells, del_but)
                i += 1
            for var_out in c.var_out:
                inner_analysis_out(c, var_out, i, group_cells, del_but)
                i += 1

            for n in range(len(group_cells)):
                for m in range(n + 1, len(group_cells)):
                    if group_cells[n][0] == group_cells[m][0]:
                        for p in range(4):
                            if (
                                group_cells[n][1][p].read_only is False
                                and group_cells[m][1][p].read_only is False
                            ):
                                widgets.jslink(
                                    (group_cells[n][1][p], "value"),
                                    (group_cells[m][1][p], "value"),
                                )

    analyse_but.loading = False
    analyse_but.children = ["Analysis done"]
    function.background_color = "#D9D9D9"
    pack_area.background_color = "#D9D9D9"
    vb.children = list(vb.children) + [sheet, derivative_check, print_but, gen_but]

    global RESULT
    RESULT = result

    global HG_DATA
    HG_DATA = hg_data


def print_click(widget, event, data):
    # Function called when the button "print code" is pressed

    print_but.loading = True
    print_but.disabled = True

    sheet = ipysheet.current()
    arr = ipysheet.to_array(sheet)

    global RESULT
    result = RESULT

    global HG_DATA
    hg_data = HG_DATA

    global DELETED_VAR
    deleted_var = DELETED_VAR
    deleted_var.sort(reverse=True)

    global GEN_OR_PRINT

    global PACK

    global S
    global LS

    if not GEN_OR_PRINT:
        sheet.read_only = True
        # Modifying result to take into account changes made to the sheet
        i = 0
        for g in result:
            for c in g[1]:
                for var_in in c.var_in:
                    var_in.symbol = arr[i, 2]
                    var_in.name = arr[i, 3]
                    var_in.unit = arr[i, 5]
                    var_in.val = arr[i, 6]
                    i += 1
                for var_out in c.var_out:
                    var_out.symbol = arr[i, 2]
                    var_out.name = arr[i, 3]
                    var_out.unit = arr[i, 5]
                    i += 1

        # Delete deleted variables
        for index in deleted_var:
            delete_var(result, index)

        vb.children = list(vb.children) + [n2_but]
        S, LS = gen_str.multi_rec_gen_string(hg_data, PACK, derivative_check.v_model)
        if not LS:
            n2_but.children = ["cannot generate n2 diagram (groups only)"]
        else:
            n2_but.disabled = False

    GEN_OR_PRINT = True

    # noinspection PyTypeChecker
    display(Markdown("```python\n" + S + "\n```"))

    print_but.loading = False
    print_but.children = ["Code printed"]


def gen_click(widget, event, data):
    # Function called when the button "generate file" is pressed
    gen_but.loading = True
    gen_but.disabled = True

    sheet = ipysheet.current()
    arr = ipysheet.to_array(sheet)

    global RESULT
    result = RESULT

    global HG_DATA
    hg_data = HG_DATA

    global DELETED_VAR
    deleted_var = DELETED_VAR
    deleted_var.sort(reverse=True)

    global GEN_OR_PRINT

    global PACK

    if not GEN_OR_PRINT:
        sheet.read_only = True
        # Modifying result to take into account changes made to the sheet
        i = 0
        for g in result:
            for c in g[1]:
                for var_in in c.var_in:
                    var_in.symbol = arr[i, 2]
                    var_in.name = arr[i, 3]
                    var_in.unit = arr[i, 5]
                    var_in.val = arr[i, 6]
                    i += 1
                for var_out in c.var_out:
                    var_out.symbol = arr[i, 2]
                    var_out.name = arr[i, 3]
                    var_out.unit = arr[i, 5]
                    i += 1

        # Delete deleted variables
        for index in deleted_var:
            delete_var(result, index)

        vb.children = list(vb.children) + [n2_but]
        global S
        global LS
        S, LS = gen_str.multi_rec_gen_string(hg_data, PACK, derivative_check.v_model)
        if not LS:
            n2_but.children = ["cannot generate n2 diagram (groups only)"]
        else:
            n2_but.disabled = False

    GEN_OR_PRINT = True

    new_generate_file(hg_data, PACK, derivative_check.v_model)

    gen_but.loading = False
    gen_but.children = ["File generated "]


def n2_click(widget, event, data):
    # Function called when the button "generate n2 diagram" is pressed
    n2_but.loading = True
    global LS
    global HG_DATA
    n2_globals.set_globals(LS, HG_DATA)
    importlib.reload(n2_execute)
    n2_but.loading = False
    n2_but.disabled = True
    n2_but.children = ["N2 diagram generated"]
