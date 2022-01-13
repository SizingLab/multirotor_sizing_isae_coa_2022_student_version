from . import variable_recognition as vr
from . import parse_pack as pp

TEXT = (
    "x = y3**2.5+4e-10 + np.exp + mat.exp + math.exp + numpy.exp #[m]\n"
    "b = Dext/(Dalpha*1.0e-1 + 1) #[kg]\n"
    "force = np.log(np.cos(lift + 1.3e+4)+sin(np.log(np.exp(x)) + 1\n"
    "if area == 3e+3:\n"
    "    elif j_2 < 3:\n"
    "        j_2 += 1.1e+2 + 2\n"
)

TEST = (
    "ratio_x025 = x0_25 / fus_length\n"
    "k_h = 0.01222 - 7.40541e-4 * ratio_x025 * 100 + 2.1956e-5 * (ratio_x025 * 100) ** 2"
)
PACK = "numpy as np, math as mat"


def main():

    inp, out = vr.get_variables(TEXT, pp.parse_pack(PACK))
    inputs = []
    outputs = []
    print("Inputs detected:")
    print([var_in.symbol for var_in in inp])
    print("Outputs detected:")
    print([var_out.symbol for var_out in out])
    print("Units detected:")
    print([var_out.unit for var_out in out])
    print("First output:")
    print(out[0])
    for x in inp:
        x.name = input("input name for {}:".format(x.symbol))
        inputs.append(x)
    for x in out:
        x.name = input("output name for {}:".format(x.symbol))
        outputs.append(x)
    text = TEXT
    print("--edited function--")
    print(vr.edit_function(inputs, outputs, text))


if __name__ == "__main__":
    main()
