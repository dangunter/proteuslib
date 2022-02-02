"""
Interpreter for DSL
"""
import builtins
import re
from dataclasses import dataclass
from IPython.core.magic import Magics, magics_class, cell_magic, line_magic

#
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


class DSLSyntaxError(Exception):
    def __init__(self, args=None, action="unknown", why="Unknown reason"):
        cmd = " ".join(args) if args is not None else "unknown"
        msg = f"In '{action} {cmd}': {why}"
        super().__init__(self, msg)
        self.message = msg


def interpreter():
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
        set port flow mass $phase:liq|vap $value:f
    """
    _VAR = "$"
    # Get builtin keywords (lowercase, no dunders)
    _BI = [s for s in dir(builtins) if not s[0] == "_" and s.lower() == s]
    # Functions to parse different type codes
    _PARSE_FN = {
        "s": str,
        "f": float,
        "i": int,
        "d": int
    }

    def __init__(self):
        self._command_patterns = []
        self._state = {"_interp": self}

    def get_state(self, name):
        return self._state.get(name, None)

    def add_command(self, pattern, func):
        tokens = self._tokenize(pattern)
        # Get parse functions for the variables
        variables = {}
        for t in tokens:
            if t.startswith(self._VAR):
                var_name, parse_func = self._parse_variable(t)
                variables[t] = (var_name, parse_func)
        print(f"@@ variables for '{pattern}' => {variables}")
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
        func(state=self._state, **kwargs)

    def match_command(self, text):
        tokens = self._tokenize(text)  # should be at least 1 token
        if tokens[0] in ("#", "!"):
            return self._comment, {}
        kwargs = {}
        for pattern_tokens, variables, func in self._command_patterns:
            print(f"@@ pattern-tokens={pattern_tokens} tokens={tokens}")
            if len(pattern_tokens) != len(tokens):
                continue
            success = True
            for tt, pt in zip(tokens, pattern_tokens):
                print(f"@@   tt={tt} pt={pt}")
                # If a variable value is expected, parse it
                if pt in variables:
                    print("@@     is-variable")
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


# set inlet flow mass NaCl 0.035
# set inlet flow mass H2O 0.965
# set inlet pressure 50e5
# set inlet temperature 298.15
# set membrane area 50
# set membrane permeability water 4.2e-12
# set membrane permeability salt 3.5e-8
# set permeate pressure 101325

def _add_standard_commands(interp):
    cmd = StandardCommands
    interp.add_command("init", cmd.create_model)
    interp.add_command("unit ro-0d", cmd.ro_0d)


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


class RO0DCommands:

    @classmethod
    def add_all(cls, interp):
        interp.add_command("set inlet flow $type:mass|mol $comp:NaCl|H2O $value:f",
                           cls.inlet_flow_phase)
        interp.add_command("set inlet pressure $value:f", cls.inlet_pressure)
        interp.add_command("set inlet temperature $value:f", cls.inlet_temperature)

    @staticmethod
    def inlet_flow_phase(state=None, comp=None, value=None, type_=None):
        m = state["model"]
        phase = "Liq"
        if type_ == "mass":
            m.fs.unit.inlet.flow_mass_phase_comp[0, phase, comp].fix(value)
        elif type_ == "mol":
            m.fs.unit.inlet.flow_mol_phase_comp[0, phase, comp].fix(value)
        # else: ??

    @staticmethod
    def inlet_pressure(state=None, value=None):
        m = state["model"]
        m.fs.unit.inlet.pressure[0].fix(value)

    @staticmethod
    def inlet_temperature(state=None, value=None):
        m = state["model"]
        m.fs.unit.inlet.temperature[0].fix(value)

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


#####################################################################


class OldInterpreter:
    def __init__(self):
        self._commands = {}
        self._model = create_model()
        self._state = State(model=self._model, unit=None, results=None, scaled=False,
                            _interp=self)

    def add_command(self, command_class: type) -> 'Command':
        command_obj = command_class(self._state)
        self._commands[command_class.name] = command_obj
        return command_obj

    def get_command(self, name) -> 'Command':
        return self._commands[name]

    def process(self, text):
        for line in text.split("\n"):
            tokens = line.strip().split()
            if len(tokens) < 1:
                continue
            command_name = tokens[0]
            if command_name in ("#", "!"):
                print(f"Comment: {' '.join(tokens[1:])}")
                continue
            if command_name == _help_keyword:
                lines = ["IDAES commands to create, configure, and run models.",
                         f"Add '{_help_keyword}' after any command to see options.",
                         "Commands:"]
                for name, cmd in self._commands.items():
                    lines.append(f"  {name}: {cmd.description()}")
                display_help(lines)
                continue
            if command_name not in self._commands:
                print(f"Unknown command: {command_name}")
                continue
            command = self._commands[command_name]
            try:
                command.run(tokens[1:])
            except DSLSyntaxError as err:
                print(f"Syntax error: {err.message}")
                continue
            # Do not print any reports of actions if the user is getting help
            # (eventually; not for this command)
            if tokens[-1] != _help_keyword:
                print(f"Action: {' '.join(command.get_report())}")

    @property
    def state(self):
        return self._state


@dataclass
class State:
    model: object
    unit: object
    results: object
    scaled: bool
    _interp: Interpreter


class Command:
    name = "<base>"

    def __init__(self, state: State):
        self._state = state
        self._subcommands = {}
        self._report = ()
        self._parent = None

    def add_command(self, command_class: type) -> "Command":
        command_obj = command_class(self._state)
        command_obj.parent = self
        self._subcommands[command_obj.name] = command_obj
        return command_obj

    def get_report(self):
        return self._report

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        if self._parent is not None:
            raise RuntimeError("Cannot set parent command more than once")
        self._parent = value

    def run(self, args):
        self._report = ()
        command = None
        # print(f"@@ args={args}")
        if len(args) > 0 and args[0] == _help_keyword:
            lines = self.help()
            display_help(lines)
        elif len(args) > 0 and self._subcommands:
            subcommand_name = args[0].lower()
            new_args = self._execute(args)
            # allow modification of args, but do nothing if return is None
            if new_args is not None:
                args = new_args
            try:
                command = self._subcommands[subcommand_name]
            except KeyError:
                self.syntax_error(
                    args, f"unknown subcommand for '{self.name}': " f"{subcommand_name}"
                )
            args = args[1:]
            command.run(args)
            if self.get_report():
                self._report = (
                    list(self.get_report()) + ["+"] + list(command.get_report())
                )
            else:
                self._report = command.get_report()
        else:
            self._execute(args)

    def description(self):
        return getattr(self, "desc", self.__class__.__name__)

    def help(self):
        lines = [self.description()]
        if len(self._subcommands) > 0:
            lines.append("Subcommands:")
            for name, subc in self._subcommands.items():
                lines.append(f"  {name}: {subc.description()}")
        return lines

    def _execute(self, args):
        pass

    def syntax_error(self, args, error):
        raise DSLSyntaxError(args=args, action=self.name, why=error)


class UnitCommand(Command):
    name = "unit"
    desc = "Create unit"

    def _execute(self, args):
        if self._state.unit is not None:
            self.syntax_error(args, "Cannot create another unit")


class ClearCommand(Command):
    name = "clear"
    desc = "Clear and reset the model"

    def _execute(self, args):
        self._state.model = create_model()
        self._state.unit = None
        self._report = ("clear model",)


class SetCommand(Command):
    name = "set"
    desc = "Set (fix) a value"

    def _execute(self, args):
        val, float_val = None, None
        try:
            val = args[-1]
        except IndexError:
            self.syntax_error(args, "Missing value for 'set'")
        if val == _help_keyword:
            return
        try:
            float_val = float(val)
        except ValueError:
            self.syntax_error(args, f"Cannot convert value '{val}' to a number")
        self._state.set_val = float_val
        return args[:-1]


class SetPort(Command):
    name = "<port>"

    def _execute(self, args):
        self._state.set_port = self._get_unit_port(self.name)
        self._state.set_what = self.name

    def _get_unit_port(self, what):
        """Retrieve the corresponding port object from the model."""
        try:
            obj = getattr(self._state.model.fs.unit, what)
        except AttributeError:
            raise ValueError(f"Cannot find port {what} for unit")
        return obj

    def _set_indexed_value(self, attr, index, args):
        port, val = self._state.set_port, self._state.set_val
        try:
            prop = getattr(port, attr)
        except AttributeError:
            raise ValueError(f"Unknown property: {attr}")
        try:
            prop[index].fix(val)
        except KeyError as err:
            self.syntax_error(args, f"Error fixing indexed property value: {err}")


class SetInlet(SetPort):
    """An inlet is a port with the name 'inlet'.
    """

    name = "inlet"
    desc = "Set an inlet parameter"


class SetOutlet(SetPort):
    """An outlet is a port with the name 'outlet'.
    """

    name = "outlet"
    desc = "Set an outlet parameter"


class SetPortFlow(SetPort):

    name = "flow"
    desc = "Set flow rate for an inlet or outlet"

    def _execute(self, args):
        """Override `SetPort._execute` to do nothing, instead."""
        pass


class SetPortFlowMassPhase(SetPort):
    """Shared functionality for SetPortFLowMass<Liq,Vap> subclasses."""
    def _execute_phase(self, args, phase):
        if len(args) < 1:
            self.syntax_error(args, "Missing <component> name")
        elif len(args) > 1:
            self.syntax_error(args, "Extra arguments after <component> name")
        comp = args[0]
        self._set_indexed_value("flow_mass_phase_comp", (0, phase, comp), args)
        self._report = ("fix", self._state.set_what, "flow", f"mass ({phase})", comp,
                        f"<- {self._state.set_val}")

    def help(self):
        """Help used by Liq and Vap subclasses.
        """
        what = self._state.set_what
        return [self.desc,
                f"Syntax: set {what} flow mass <component> value",
                f"  <component> = chemical formula like 'NaCl' or 'H2O'"]


class SetPortFlowMassLiq(SetPortFlowMassPhase):

    name = "mass"
    desc = "Set mass flow rate for liquid phase at an inlet or outlet"

    def _execute(self, args):
        return self._execute_phase(args, "Liq")


class SetPortFlowMassVap(SetPortFlowMassPhase):

    name = "mass"
    desc = "Set mass flow rate for vapor phase at an inlet or outlet"

    def _execute(self, args):
        return self._execute_phase(args, "Vap")


class SetPortFlowScalar(SetPort):
    """Base class for setting simple scalar values.
    Use a subclass for the command.
    """
    def _execute(self, args):
        attr = self.name
        self._set_indexed_value(attr, (0,), args)
        self._report = ("fix", self._state.set_what, attr, f"<- {self._state.set_val}")


class SetPortFlowPressure(SetPortFlowScalar):
    name = "pressure"
    desc = "Set the flow pressure"


class SetPortFlowTemperature(SetPortFlowScalar):
    name = "temperature"
    desc = "Set the flow temperature"


class SetPermeate(Command):
    name = "permeate"
    desc = "Set properties for the permeate"


class SetPermeateScalar(Command):
    """Base class for setting simple scalar values.
    Use a subclass for the command.
    """
    def _execute(self, args):
        prop = getattr(self._state.unit.permeate, self.name)[0]
        prop.fix(self._state.set_val)
        self._report = ("fix", "permeate", self.name, f"<- {self._state.set_val}")

    def help(self):
        return [self.desc, f"Syntax: set permeate {self.name} <value>"]


class SetPermeatePressure(SetPermeateScalar):
    name = "pressure"
    desc = "Set the permeate pressure"


class SetMembrane(Command):
    name = "membrane"
    desc = "Set properties for the membrane"


class SetMembraneArea(Command):
    name = "area"
    desc = "Set membrane area (m^2)"

    def _execute(self, args):
        self._state.unit.area.fix(self._state.set_val)
        self._report = ("fix membrane area", f"<- {self._state.set_val}")


class SetMembranePermeability(Command):
    name = "permeability"
    desc = "Set membrance permeability for a given component"

    def _execute(self, args):
        try:
            permeate = args[0].lower()
        except IndexError:
            self.syntax_error(args, "Missing name of permeate, 'water' or 'salt'")
        try:
            comp = {"water": "A", "salt": "B"}[permeate]
        except KeyError:
            self.syntax_error(args, "Wrong name of permeate, 'water' or 'salt'")
        getattr(self._state.unit, f"{comp}_comp").fix(self._state.set_val)
        self._report = ("fix", "membrane", permeate, "permeability",
                        f"<- {self._state.set_val}")

    def help(self):
        return [self.desc, "Syntax: set membrane permeability (water|salt) <value>"]


class RO0DUnit(Command):
    desc = "Reverse osmosis 0-D unit operation"
    name = "ro-0d"

    def _execute(self, args):
        m = self._state.model
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
        self._state.unit = unit
        # add commands to set values on the unit
        set_cmd = g_interpreter.get_command("set")
        # create a 'set inlet' parent command
        inlet_cmd = set_cmd.add_command(SetInlet)
        # add: set inlet flow ...
        inlet_cmd.add_command(SetPortFlow).add_command(SetPortFlowMassLiq)
        inlet_cmd.add_command(SetPortFlowPressure)
        inlet_cmd.add_command(SetPortFlowTemperature)
        set_cmd.add_command(SetPermeate).add_command(SetPermeatePressure)
        mbr_cmd = set_cmd.add_command(SetMembrane)
        mbr_cmd.add_command(SetMembraneArea)
        mbr_cmd.add_command(SetMembranePermeability)

        self._report = ("unit", "RO-0D")


def load_commands(interp):
    # clear
    interp.add_command(ClearCommand)
    # unit command
    unit_cmd = interp.add_command(UnitCommand)
    unit_cmd.add_command(RO0DUnit)
    # set command
    set_cmd = interp.add_command(SetCommand)
    # add set, scale, init, solve, etc.


def interp(s, model=None):
    if model is None:
        m = ConcreteModel()
        m.fs = FlowsheetBlock(default={"dynamic": False})
    else:
        m = model

    state = State(model=m, unit=getattr(m.fs, "unit", None), results=None, scaled=False)

    for line in s.split("\n"):
        tokens = line.strip().split()
        if len(tokens) < 1:
            continue
        action_token = tokens[0]
        action = Action(action_token)
        if action.is_valid():
            try:
                what_was_done = perform_action(action, tokens[1:], state)
            except DSLSyntaxError as err:
                print(f"SyntaxError: {err.message}")
                return None
            print(f"Action: {' '.join(what_was_done)}")
        else:
            print(f"Unknown action: {action_token}")

    return state


def is_match(expr, text):
    m = re.match(expr, text)
    return m is not None


def perform_action(action, args, state):
    # print(f"@@ STATE: {state}")

    def syntax_error(reason, action=action, args=args):
        raise DSLSyntaxError(args=args, action=action.name(), why=reason)

    m = state["model"]
    summary = ()
    if action.unit():
        if state["unit"] is not None:
            raise DSLSyntaxError(
                args=args, action=action.name(), why="Model unit already defined"
            )
        unit_type = args[0].lower()
        if is_match(r"ro-*0d", unit_type):
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
            state["unit"] = unit
            summary = ("unit", unit_type)
        else:
            raise ValueError(f"Unknown unit type: {unit_type}")
    elif action.fix():
        if len(args) < 2:
            syntax_error("Need at least a target and value")
        what = args[0].lower()
        value = args[-1]
        try:
            float_val = float(value)
        except ValueError:
            syntax_error(f"Cannot convert value '{value}' to a number")
        if what in ("inlet", "outlet"):
            if len(args) < 3:
                syntax_error("Need more arguments for setting {what} property")
            try:
                port = getattr(m.fs.unit, what)
            except AttributeError:
                raise ValueError(f"Cannot find {what} for unit")
            parts = args[1:-1]
            set_prop = parts[0].lower()
            if set_prop == "flow":
                # set inlet flow mass NaCl 0.035 ->
                # m.fs.unit.inlet.flow_mass_phase_comp[0, 'Liq', 'NaCl'].fix(0.035)
                if len(parts) != 3:
                    syntax_error(
                        f"Wrong arguments for 'set {what} flow', "
                        f"should be: set {what} flow <mass> <component> value"
                    )
                flow_type, comp_name = parts[1], parts[2]
                attr = f"flow_{flow_type}_phase_comp"
                phase = "Liq"
                prop = getattr(port, attr, None)
                if prop is None:
                    raise ValueError(f"Unknown property: {attr}")
                prop[0, phase, comp_name].fix(float_val)
                summary = ("fix", what, "flow", flow_type, comp_name, f"<- {float_val}")
            elif set_prop in ("pressure", "temperature"):
                # set inlet pressure 50e5 ->
                # m.fs.unit.inlet.pressure[0].fix(50e5)
                if len(parts) > 1:
                    syntax_error(f"Too many arguments for 'set {what} {set_prop}'")
                prop = getattr(port, set_prop, None)
                if prop is None:
                    raise ValueError(f"Unkown {what} property: {set_prop}")
                prop[0].fix(float_val)
                summary = ("fix", what, set_prop, f"<- {float_val}")
            else:
                syntax_error(f"Unknown inlet property '{inlet_prop}'")
        elif what == "membrane":
            # set membrane area 50
            # set membrane permeability water 4.2e-12
            # set membrane permeability salt 3.5e-8 ->
            # m.fs.unit.area.fix(50)# membrane area (m^2)
            # m.fs.unit.A_comp.fix(4.2e-12) membrane water permeability (m/Pa/s)
            # m.fs.unit.B_comp.fix(3.5e-8)# membrane salt permeability (m/s)
            if len(args) < 3:
                syntax_error(
                    "Wrong arguments for 'set membrane', should be "
                    "'set membrane <property> [<component>] <value>'"
                )
            parts = args[1:-1]
            set_prop = parts[0].lower()
            if set_prop == "area":
                if len(parts) > 1:
                    syntax_error("Too many arguments for 'set membrane area'")
                m.fs.unit.area.fix(float_val)
                summary = ("fix", what, set_prop, f"<- {float_val}")
            elif set_prop == "permeability":
                if len(parts) != 2:
                    syntax_error(
                        f"Wrong number of arguments for 'set membrane {set_prop}', should be "
                        f"'set membrane {set_prop} <salt|water> <value>'"
                    )
                comp = parts[1].lower()
                if comp == "water":
                    comp_attr = "A_comp"
                elif comp == "salt":
                    comp_attr = "B_comp"
                else:
                    raise ValueError(
                        f"Unknown membrane {set_prop} component '{comp}', should be "
                        "'water' or 'salt'"
                    )
                getattr(m.fs.unit, comp_attr).fix(float_val)
                summary = ("fix", what, set_prop, f"(comp)", f"<- {float_val}")
            else:
                raise ValueError(f"Unknown membrane property: {set_prop}")
        elif what == "permeate":
            # set permeate pressure 101325 ->
            # m.fs.unit.permeate.pressure[0].fix(101325)
            if len(args) != 3:
                syntax_error(
                    "Wrong arguments for 'set permeate', should be "
                    "'set permeate <property> <value>'"
                )
            set_prop = args[1].lower()
            if set_prop in ("pressure", "temperature"):
                getattr(m.fs.unit.permeate, set_prop)[0].fix(float_val)
                summary = ("fix", what, set_prop, f"<- {float_val}")
            else:
                raise ValueError("Unknown permeate property: {set_prop}")
        else:
            syntax_error(f"Unknown target: {what}")
    elif action.scale():
        # m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1, index=('Liq', 'H2O'))
        # m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1e2, index=('Liq', 'NaCl')) ->
        # scale flow mass NaCl 1e2
        # scale flow mass H2O 1
        if len(args) < 2:
            syntax_error("Need at least a subject and value")
        what = args[0].lower()
        value = args[-1]
        try:
            float_val = float(value)
        except ValueError:
            syntax_error(f"Cannot convert value '{value}' to a number")
        if what == "flow":
            if len(args) != 4:
                syntax_error(
                    "Wrong number of arguments for 'scale flow', should be "
                    "scale flow <mass|volume> <component> <value>"
                )
            attr = f"flow_{args[1]}_phase_comp"
            index = ("Liq", args[2])
            m.fs.properties.set_default_scaling(attr, float_val, index=index)
            summary = ("scale", what, args[1], str(index), f"<- {float_val}")
        else:
            raise ValueError(f"Unknown property for scaling: {what}")
        state["scaled"] = True
    elif action.init():
        if state["scaled"]:
            calculate_scaling_factors(m)
        m.fs.unit.initialize()
        summary = ("initialize", "unit")
    elif action.solve():
        solver = get_solver(options={"nlp_scaling_method": "user-scaling"})
        state["results"] = solver.solve(m)
        summary = ("solve", "unit", "scaling=user")
    elif action.noop():
        pass

    return summary


class Action:
    _noop = "noop"
    _unit = "unit"
    _fix = "fix"
    _set = _fix
    _scale = "scale"
    _init = "initialize"
    _solve = "solve"

    def __init__(self, tok):
        tok = tok.lower()
        if tok.startswith("_"):
            a = None
        elif tok[0] in ("#", "!"):
            a = self._noop
        else:
            a = getattr(self, "_" + tok, None)
        self._value = a

    # actions
    def unit(self):
        return self._value == self._unit

    def fix(self):
        return self._value == self._fix

    def scale(self):
        return self._value == self._scale

    def init(self):
        return self._value == self._init

    def solve(self):
        return self._value == self._solve

    def noop(self):
        return self._value == self._noop

    # name
    def name(self):
        if not self.is_valid():
            return "(invalid)"
        return self._value

    # validity
    def is_valid(self):
        return self._value is not None



