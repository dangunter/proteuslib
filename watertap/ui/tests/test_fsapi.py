"""
Tests for fsapi module
"""
import logging

import pytest

from pyomo.environ import units as pyunits
from pyomo.environ import Var

from watertap.examples.flowsheets.case_studies.seawater_RO_desalination import (
    seawater_RO_desalination as RO,
)
from watertap.ui import fsapi


_log = logging.getLogger("idaes.watertap.ui.fsapi")
_log.setLevel(logging.DEBUG)


def build_ro(erd_type=None):
    model = RO.build_flowsheet(erd_type=erd_type)
    return model.fs


def solve_ro(flowsheet=None):
    assert flowsheet


class InputCategory:
    """Names for input categories"""

    feed = "Feed"
    hydrogen = "Hydrogen reactor"
    methane = "Methane reactor"
    system = "System parameters"


class OutputCategory:
    """Names for output categories"""

    feed = "Feed"
    levelized_costs = "Levelized costs"
    normalized_costs = "Normalized costs"
    normalized_performance = "Normalized performance"
    product = "Product"
    capital_cost = "Capital cost"
    operating_cost = "Operating cost"
    revenue = "Revenue"


def export_to_ui(flowsheet=None, exports=None):
    fs = flowsheet
    exports.add(
        obj=fs.feed.flow_vol[0],
        name="Flowrate",
        ui_units=pyunits.m**3 / pyunits.hr,
        display_rounding=1e-2,  # round to nearest 0.01
        description="Volumetric flowrate for the feed",
        is_input=True,
        input_category=InputCategory.feed,
        read_only=False,
        is_output=True,
        output_category=OutputCategory.feed,
    )


def flowsheet_interface(exports=True):
    kwargs = {}
    if exports:
        kwargs["do_export"] = export_to_ui
    return fsapi.FlowsheetInterface(
        # leave out name and description to test auto-fill
        do_build=build_ro,
        do_solve=solve_ro,
        **kwargs
    )


@pytest.mark.unit
def test_create_interface():
    fsi = flowsheet_interface(exports=False)
    fsi = flowsheet_interface()
    fs2 = fsapi.FlowsheetInterface(fs=fsi.fs_exp)
    assert fs2.fs_exp == fsi.fs_exp


@pytest.mark.unit
def test_build_noexport():
    fsi = flowsheet_interface(exports=False)
    with pytest.raises(KeyError):
        fsi.build(erd_type="pressure_exchanger")


@pytest.mark.unit
def test_build():
    fsi = flowsheet_interface()
    fsi.build(erd_type="pressure_exchanger")
    data = fsi.dict()
    print(data)
    assert "model_objects" in data
    assert len(data["model_objects"]) == 1


@pytest.mark.unit
def test_actions():
    fsi = flowsheet_interface()
    built = False
    garbage = {"trash": True}
    v1 = Var(name="variable1")
    v1.construct()
    v1.value = 1
    print(v1.display())

    def fake_build():
        nonlocal built
        built = True
        return garbage

    def fake_solve(flowsheet=None):
        # flowsheet passed in here should be what fake_build() returns
        assert flowsheet == garbage

    def fake_export(flowsheet=None, exports=None):
        with pytest.raises(AttributeError):
            exports.add(obj=garbage)
        exports.add(obj=v1)  # form 1
        exports.add(v1)  # form 2
        exports.add(data=v1)  # form 3
        ve1 = fsapi.ModelExport(obj=v1)
        exports.add(data=ve1.dict())  # form 4
        with pytest.raises(ValueError):
            exports.add(v1, v1)

    fsi.add_action(fsapi.Actions.build, fake_build)
    fsi.add_action(fsapi.Actions.export, fake_export)
    fsi.add_action(fsapi.Actions.solve, fake_solve)
    fsi.build()
    fsi.solve()
    with pytest.raises(ValueError):
        fsi.run_action(fsapi.Actions.export)


@pytest.mark.unit
def test_load():
    fsi = flowsheet_interface()
    fsi.build(erd_type="pressure_exchanger")
    # get some info
    var_key = list(fsi.fs_exp.model_objects.keys())[0]
    var_obj = fsi.fs_exp.model_objects[var_key].obj
    save_value = var_obj.value
    # serialize
    data = fsi.dict()
    # modify
    data["model_objects"][var_key]["value"] = -1000
    # reload
    fsi.load(data)
    # check
    assert fsi.fs_exp.model_objects[var_key].value == -1000

    # this time with a missing thing
    data = fsi.dict()
    # add another (fake) one
    data["model_objects"]["foobar"] = data["model_objects"][var_key].copy()
    # reload (fake one will be 'missing')
    with pytest.raises(fsapi.MissingObjectError):
        fsi.load(data)


@pytest.mark.unit
def test_find_smoke():
    fsapi.FlowsheetInterface.find("watertap")


@pytest.mark.component
def test_find():
    result = fsapi.FlowsheetInterface.find("examples.ui")
    assert len(result) == 1  # expect only 1 module
    func = list(result.values())[0]  # get the module function
    interface = func()  # invoke the function to get the interface object
    interface.build()  # make sure the module exported properly
    with pytest.raises(KeyError):
        interface.solve()
