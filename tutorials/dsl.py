"""
Interpreter for DSL
"""
import builtins
import re
from dataclasses import dataclass
from IPython.core.magic import Magics, magics_class, cell_magic, line_magic

# Imports from Pyomo, including "value" for getting the
# value of Pyomo objects
from pyomo.environ import ConcreteModel, Objective, Expression, value

# Imports from IDAES
# Import flowsheet block from IDAES core
from idaes.core import FlowsheetBlock

# Import function to get default solver
from idaes.core.util import get_solver

# Import function to check degrees of freedom
from idaes.core.util.model_statistics import degrees_of_freedom

# Import utility function for calculating scaling factors
from idaes.core.util.scaling import calculate_scaling_factors

# Imports from WaterTAP
# Import NaCl property model
from watertap.property_models.NaCl_prop_pack import NaClParameterBlock

# Import RO model
from watertap.unit_models.reverse_osmosis_0D import (
    ReverseOsmosis0D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
)


def interpreter() -> "Interpreter":
    """Create and return a new interpreter instance."""
    interp = Interpreter()
    _add_standard_commands(interp)
    return interp


class Interpreter:
    """A simple command interpreter driven by a set of patterns.

    These are space-separated tokens. A 'bare' token represents what the user
    must type in the command. A '$' prefix means a variable.

      $name => String variable 'name'
      $name:f => Numeric (floating-point) variable 'name'
      $name:foo|bar => String variable 'name' that must take value 'foo' or 'bar'

    Examples:

        pattern: unit $type
        input: unit ro-0d
        func: def unit(t): ..
        .
        fix port flow mass $phase:liq|vap $value:f
    """

    _VAR = "$"
    # Get builtin keywords (lowercase, no dunders)
    _BI = [s for s in dir(builtins) if not s[0] == "_" and s.lower() == s]
    # Functions to parse different type codes
    _PARSE_FN = {"s": str, "f": float, "i": int, "d": int}

    def __init__(self):
        self._command_patterns = []
        self._state = {"_interp": self}

    def get_state(self, name):
        return self._state.get(name, None)

    def set_default(self, name, value):
        self._state[name] = value

    def add_command(self, pattern, func):
        tokens = self._tokenize(pattern)
        # Get parse functions for the variables
        variables = {}
        for t in tokens:
            if t.startswith(self._VAR):
                var_name, parse_func = self._parse_variable(t)
                variables[t] = (var_name, parse_func)
        # print(f"@@ variables for '{pattern}' => {variables}")
        # Save the command
        self._command_patterns.append((tokens, variables, func))

    def run_commands(self, text):
        for line in text.split("\n"):
            line = line.strip()
            if line:
                self.run_command(line)

    def run_command(self, line):
        func, kwargs = self.match_command(line)
        if func is None:
            raise KeyError(f"Unknown command: '{line}'")
        # invoke function
        # print(f"Running: {line}")
        func(state=self._state, **kwargs)

    def match_command(self, text):
        tokens = self._tokenize(text)  # should be at least 1 token
        if tokens[0] in ("#", "!"):
            return self._comment, {}
        kwargs = {}
        for pattern_tokens, variables, func in self._command_patterns:
            # print(f"@@ pattern-tokens={pattern_tokens} tokens={tokens}")
            if len(pattern_tokens) != len(tokens):
                continue
            success = True
            for tt, pt in zip(tokens, pattern_tokens):
                # print(f"@@   tt={tt} pt={pt}")
                # If a variable value is expected, parse it
                if pt in variables:
                    # print("@@     is-variable")
                    var_name, parse_func = variables[pt]
                    try:
                        var_value = parse_func(tt)
                    except ValueError as err:
                        success = False
                        break
                    # append a "_" to a built-in name like 'type'
                    if var_name in self._BI:
                        var_name = var_name + "_"
                    kwargs[var_name] = var_value
                # Otherwise, expect an exact match
                elif tt.lower() != pt:
                    success = False
                    break
            if success:  # match
                return func, kwargs
        return None, None  # no match

    @staticmethod
    def _tokenize(s):
        return s.strip().split()

    @classmethod
    def _parse_variable(cls, pattern_token):
        parts = pattern_token.split(":", 1)
        # no type => string
        if len(parts) == 1:
            var_name = parts[0][1:]
            return var_name, str
        raw_var_name, var_type = parts
        var_name = raw_var_name[1:].lower()  # strip leading variable-symbol
        # if '|' in type => enum
        if "|" in var_type:
            allowed = var_type.split("|")

            def fn(a):
                def _fn(tok):
                    if tok in a:
                        return tok
                    raise ValueError(f"Input '{tok}' not in: {','.join(a)}")

                return _fn

            return var_name, fn(allowed)
        # otherwise lookup in class mapping
        try:
            var_parse_func = cls._PARSE_FN[var_type]
        except KeyError:
            raise ValueError(f"Unknown type code '{var_type}' in '{pattern_token}'")
        return var_name, var_parse_func

    @staticmethod
    def _comment(state=None):
        """No-op function for comments"""
        return


def _add_standard_commands(interp: Interpreter):
    cmd = StandardCommands
    # built-in commands
    interp.add_command("start", cmd.create_model)
    interp.add_command("unit ro-0d", cmd.ro_0d)
    interp.add_command("init", cmd.init)
    interp.add_command("solve", cmd.solve)
    # default values
    interp.set_default("scaled", False)


class StandardCommands:
    @staticmethod
    def create_model(state=None):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(default={"dynamic": False})
        state["model"] = m

    @staticmethod
    def ro_0d(state=None):
        m = state["model"]
        # create the unit operation
        m.fs.properties = NaClParameterBlock()
        unit = ReverseOsmosis0D(
            default={
                "property_package": m.fs.properties,
                "concentration_polarization_type": ConcentrationPolarizationType.none,
                "mass_transfer_coefficient": MassTransferCoefficient.none,
                "has_pressure_change": False,
            }
        )
        m.fs.unit = unit
        # add commands for this unit
        RO0DCommands.add_all(state["_interp"])

    @staticmethod
    def init(state=None):
        m = state["model"]
        if state["scaled"]:
            calculate_scaling_factors(m)
        m.fs.unit.initialize()

    @staticmethod
    def solve(state=None):
        m = state["model"]
        options = {}
        if state["scaled"]:
            options["nlp_scaling_method"] = "user-scaling"
        solver = get_solver(options=options)
        state["results"] = solver.solve(m)


class UnitCommandBase:
    @classmethod
    def create_setter(cls, interp, target, func):
        for cmd, values in (
            ("fix", "$value:f"),
            ("bounds", "$lb:f $ub:f"),
            ("unfix", ""),
        ):
            interp.add_command(f"{cmd} {target} {values}", func)

    @staticmethod
    def set_unit_prop(unit_prop, lb=None, ub=None, value=None):
        if lb is not None:
            unit_prop.unfix()
            unit_prop.setlb(lb)
            unit_prop.setub(ub)
        elif value is not None:
            unit_prop.fix(value)
        else:
            unit_prop.unfix()


class RO0DCommands(UnitCommandBase):
    """Add commands for RO 0-D model.
    """

    @classmethod
    def add_all(cls, interp):
        cls.create_setter(
            interp,
            "inlet flow $type:mass|mol $comp:NaCl|H2O",
            cls.set_inlet_flow_phase,
        )
        cls.create_setter(interp, "inlet $prop:pressure|temperature", cls.set_inlet)
        cls.create_setter(interp, "membrane area", cls.set_membrane_area)
        cls.create_setter(
            interp, "membrane permeability $comp:water|salt", cls.set_membrane_perm
        )
        cls.create_setter(interp, "permeate pressure $value:f", cls.set_perm_pressure)
        interp.add_command(
            "scale flow $type:mass|mol $comp:NaCl|H2O $value:f", cls.scale_flow
        )

    @classmethod
    def set_inlet_flow_phase(cls, state=None, comp=None, type_=None, **kwargs):
        m = state["model"]
        phase = "Liq"
        if type_ == "mass":
            p = m.fs.unit.inlet.flow_mass_phase_comp[0, phase, comp]
        elif type_ == "mol":
            p = m.fs.unit.inlet.flow_mol_phase_comp[0, phase, comp]
        else:
            raise RuntimeError(f"Unknown type for inlet flow: {type_}")
        cls.set_unit_prop(p, **kwargs)

    @classmethod
    def set_inlet(cls, state=None, prop=None, **kwargs):
        cls.set_unit_prop(getattr(state["model"].fs.unit.inlet, prop)[0], **kwargs)

    @classmethod
    def set_membrane_area(cls, state=None, **kwargs):
        cls.set_unit_prop(state["model"].fs.unit.area, **kwargs)

    @classmethod
    def set_membrane_perm(cls, state=None, comp=None, **kwargs):
        comp_attr = {"water": "A_comp", "salt": "B_comp"}[comp]
        cls.set_unit_prop(getattr(state["model"].fs.unit, comp_attr), **kwargs)

    @classmethod
    def set_perm_pressure(cls, state=None, comp=None, **kwargs):
        cls.set_unit_prop(state["model"].fs.unit.permeate.pressure[0], **kwargs)

    @staticmethod
    def scale_flow(state=None, type_=None, comp=None, value=None):
        m = state["model"]
        phase = "Liq"
        attr = f"flow_{type_}_phase_comp"
        index = (phase, comp)
        m.fs.properties.set_default_scaling(attr, 1, index=index)
        state["scaled"] = True


#################################################################


g_interpreter = interpreter()


def get_model():
    return g_interpreter.get_state("model")


def get_results():
    return g_interpreter.get_state("results")


@magics_class
class IDAESInterpreter(Magics):
    def __init__(self, shell):
        super().__init__(shell)

    @cell_magic
    def iic(self, line, cell):
        """IDAES domain-specific language for the cell."""
        self._interpret(cell)

    @line_magic
    def ii(self, line):
        self._interpret(line)

    @staticmethod
    def _interpret(text):
        g_interpreter.run_commands(text)


def load_ipython_extension(ipython):
    """
    Any module file that define a function named `load_ipython_extension`
    can be loaded via `%load_ext module.path` or be configured to be
    autoloaded by IPython at startup time.
    """
    # You can register the class itself without instantiating it.  IPython will
    # call the default constructor on it.
    ipython.register_magics(IDAESInterpreter)
