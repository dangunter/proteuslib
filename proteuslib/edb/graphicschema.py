import pyparsing as pp
from pprint import pprint
import textwrap

block_comment = pp.QuotedString(quoteChar="/*", endQuoteChar="*/")
reqflag = pp.Optional("*")
defflag = pp.Optional("!")

enum_value = pp.Word(pp.alphanums + "-_/!@#$%^&*:,.~")  # printables besides ";"
enum_values = pp.delimitedList(enum_value, delim=";")
enum_ = pp.nestedExpr("{", "}", content=enum_values)
type_name = pp.Word(pp.alphas)
scalar = type_name + pp.Optional(enum_)
scalars = pp.delimitedList(scalar, delim="|")

field_name = pp.Word(pp.alphas, pp.alphanums + "_") | pp.QuotedString(quoteChar="/")
type_ = pp.Forward()
field = pp.Group(defflag + field_name + "<" + type_ + ">" + reqflag)
obj = "(" + pp.OneOrMore(field) + ")"
array = "[" + scalars + "]"
type_ <<= obj | array | scalars

schema = pp.OneOrMore(field)
schema.ignore(block_comment)


"""
/* A chemical species that is a component in a reaction */
"""

example1 = """
/* List of parameter values */
!param<(
  v<number>        /* Chemical name of the component */
  u<string>        /* Name of an individual element */
  i<number|string> /* Component type */
)>

component<(
  name<string>*
  elements<[string]>*
  type<string{solvent;solute;anion;cation;Solvent;Solute;Anion;Cation}>
  valid_phase_types<[string]>
  phase_equilibrium_form<(Vap<str> Liq<str>)>
  parameter_data<(
    mw<param>
    pressure_crit<param>
    /.*_coeff/<param>
    /.*_ref/<param>
  )>
)>
"""

example2 = """
component <( 
    name<str>*              'Component name
    elements<[string]>*     'Name of an individual element
    /.*_coeff/<param>
)>

!param<(
  v<number>        'Chemical name of the component
  u<string>        'Name of an individual element
  i<number|string> 'Component type
)>

"""

enum_example = """
enum <(
type<string{solvent;solute;anion;cation;Solvent;Solute;Anion;Cation}>
)>
"""


example_component = """
/* THis is a comment */
component<(
  name<str>* /* cool */
  elements<[string]>*
  valid_phase_types<[string]>
  type<string{solvent;solute;anion;cation;Solvent;Solute;Anion;Cation}>
  phase_equilibrium_form<(Vap<str> Liq<str>)>
  parameter_data<(
    mw<param>
    pressure_crit<param>
    /.*_coeff/<param>
    /.*_ref/<param>
  )>
)>
"""


class Union:
    def __init__(self, *args):
        self._types = self._flatten(args)

    def __str__(self):
        return "|".join([str(t) for t in self._types])

    __repr__ = __str__

    @staticmethod
    def _flatten(a):
        result = []
        for item in a:
            if isinstance(item, Union):
                result.extend(item._types)
            else:
                result.append(item)
        return result


class ScalarMeta(type):

    def __ror__(self, other):
        if issubclass(other, type):
            other = other()
        return Union(self(), other)


class Scalar(metaclass=ScalarMeta):
    def __init__(self, desc=None, enum=None):
        self._desc = desc
        self._enum = enum

    def __str__(self):
        return self.__class__.__name__.lower()

    def __ror__(self, other):
        return Union(self, other)


class String(Scalar):
    pass


class Number(Scalar):
    pass


param = {
    "v": Number,
    "u": String,
    "i": Number | String
}

dict_component = {
    "component": {
        "name": String("cool"),
        "elements": [String],
        "valid_phase_types": [String],
        "type": String(["solvent", "solute"]),
        "phase_equilibrium_form": {"Vap": String, "Liq": String},
        "parameter_data": {
            "mw": param,
            "pressure_crit": param,
            "/.*_coeff/": param,
            "/.*_ref/": param,
        },
    }
}

def show_dict():
    pprint(dict_component)

def parse():
    print("parsing example..")
    result = schema.parseString(example1)
    return result


#    for item in result:
#        print(f"@@ {''.join(item[0])}")
# print(result.dump())
