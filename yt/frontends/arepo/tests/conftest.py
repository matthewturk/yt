"""
Title: conftest.py
Purpose: Contains fixtures for loading data.
"""


# Test parameters. Format:
# {test1: {param1 : [(val1, val2,...), (id1, id2,...)], param2 : ...}, test2: ...}
test_params = {
    "test_arepo_bullet": {
        "a": [(0, 1, 2), ("0", "1", "2")],
        "d": [(None, ("sphere", ("c", (0.1, "unitary")))), ("None", "sphere")],
        "f, w": [
            (
                (("gas", "density"), None),
                (("gas", "temperature"), None),
                (("gas", "temperature"), ("gas", "density")),
                (("gas", "velocity_magnitude"), None),
            ),
            (
                "density-None",
                "temperature-None",
                "temperature-density",
                "velocity-None",
            ),
        ],
    },
    "test_arepo_tng59": {
        "a": [(0, 1, 2), ("0", "1", "2")],
        "d": [(None, ("sphere", ("c", (0.5, "unitary")))), ("None", "sphere")],
        "f, w": [
            (
                (("gas", "density"), None),
                (("gas", "temperature"), None),
                (("gas", "temperature"), ("gas", "density")),
                (("gas", "H_number_density"), None),
                (("gas", "H_p0_number_density"), None),
                (("gas", "H_p1_number_density"), None),
                (("gas", "El_number_density"), None),
                (("gas", "C_number_density"), None),
                (("gas", "velocity_magnitude"), None),
                (("gas", "magnetic_field_strength"), None),
            ),
            (
                "density-None",
                "temperature-None",
                "temperature-density",
                "H-None",
                "Hp0-None",
                "Hp1-None",
                "El-None",
                "C-None",
                "velocity-None",
                "magnetic_field_strength-None",
            ),
        ],
    },
}


def pytest_generate_tests(metafunc):
    for test_name, params in test_params.items():
        if metafunc.function.__name__ == test_name:
            for param_name, param_vals in params.items():
                metafunc.parametrize(param_name, param_vals[0], ids=param_vals[1])
