from .n2_globals import LS, HG_DATA, file_check

for i in range(len(LS)):
    if not HG_DATA.last:
        name = HG_DATA.children[i].name
    else:
        name = HG_DATA.children[i][0]
    f_name = file_check(name)
    suf = (
        "\n\np = om.Problem()"
        "\nmodel = p.model"
        '\nmodel.add_subsystem("{}", {}(), promotes=["*"])'
        "\np.setup()"
        '\nom.n2(p, outfile="{}.html")'.format(name, name, f_name)
    )
    string = LS[i] + suf
    exec(string)

# om.n2(p, outfile="{}.py", embeddable = True, show_browser=True)
