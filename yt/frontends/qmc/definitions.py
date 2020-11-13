import unyt


qmc_unit_sys = unyt.UnitSystem(
    "qmc",
    unyt.angstrom,
    unyt.amu,
    unyt.second,
)


elementRegister = {}
with open("elements.txt", "r") as fd:
    for line in fd:
        # Skip comments
        if line[0] == "#":
            continue
        element, symbol, atomic_number, atomic_mass = line
        # qmc data files provide the atomic numbers of the elements
        # they contain as one of their fields, so we key the register
        # by the atomic number
        elementRegister[atomic_number.strip()] = {
            "element" : element.strip(),
            "symbol" : symbol.strip(),
            "mass" : atomic_mass.strip()
        }
