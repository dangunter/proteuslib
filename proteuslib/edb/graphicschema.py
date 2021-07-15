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
type_ <<= (obj | array | scalars)

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

def parse():
    print("parsing example..")
    result = schema.parseString(example1)
    return result
#    for item in result:
#        print(f"@@ {''.join(item[0])}")
    #print(result.dump())
