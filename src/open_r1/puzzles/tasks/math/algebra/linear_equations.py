import re

import numpy as np

from ....base_config import BaseConfig
from ....base_task import BaseTask


class LinearEquationConfig(BaseConfig):
    min_coefficient: int = -10
    max_coefficient: int = 10
    min_var_value = -10
    max_var_value = 10


class LinearEquationTask(BaseTask):
    config_class = LinearEquationConfig

    def generate_sample(self, config: LinearEquationConfig, rng: np.random.Generator):
        variable_names = ("x", "y", "z", "a", "b", "c")
        var_name = rng.choice(variable_names)
        var_coefficient = 0
        while var_coefficient == 0:
            # We can't have the variable's coefficient be 0, so keep sampling until we get a non-zero one
            var_coefficient = rng.integers(config.min_coefficient, config.max_coefficient, endpoint=True)
        constant = rng.integers(config.min_coefficient, config.max_coefficient, endpoint=True)
        while var_coefficient == 1 and constant == 0:
            # We can't have the variable's coefficient be 1 and the constant be 0, as this is a trivial equation
            # so keep rerolling until it isn't
            constant = rng.integers(config.min_coefficient, config.max_coefficient, endpoint=True)
        var_value = int(rng.integers(config.min_var_value, config.max_var_value, endpoint=True))
        rhs = var_coefficient * var_value + constant

        if constant < 0:
            equation = f"{var_coefficient}{var_name} - {-constant} = {rhs}"
        elif constant > 0:
            equation = f"{var_coefficient}{var_name} + {constant} = {rhs}"
        else:
            equation = f"{var_coefficient}{var_name} = {rhs}"

        return equation, var_value

    def verify(self, output, answer):
        # If there's only one number in the output, it's the answer
        numbers = re.findall(r"\d+", output)
        if len(numbers) == 1:
            return float(int(numbers[0]) == answer)
        # If not, look for a pattern like "x = 5" to disambiguate
        numbers = re.findall(r"=\s+(\d+)", output)
        if len(numbers) == 1:
            return float(int(numbers[0].group(1)) == answer)
        # Finally, maybe it gave the answer as a decimal, so check for that
        numbers = re.findall(r"\d+\.\d+", output)
        if len(numbers) == 1:
            return float(float(numbers[0]) == answer)
