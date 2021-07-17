from pprint import pprint


class Describable:
    def __init__(self, desc=""):
        self.desc = desc

    def __floordiv__(self, other):
        """Add a comment to a scalar.
        """
        self.desc = str(other)
        return self


class Union(Describable):

    def __init__(self, *args):
        self._types = self._flatten(args)
        super().__init__()

    def __repr__(self):
        type_list = "|".join([str(t) for t in self._types])
        if self.desc:
            s = f"{type_list} /* {self.desc} */"
            return s
        else:
            return type_list

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
    """Allow bare class to be used like an instance of a subclass.
    """
    def __or__(cls, other):
        """Allow for `<ScalarSubclass> | <OtherScalarSubclass> | ..`
        """
        if issubclass(other, type):
            other = other()
        return Union(cls(), other)

    def __floordiv__(cls, other):
        """Add a comment to a scalar.
        """
        inst = cls()
        inst.desc = other
        return inst

    def __repr__(cls):
        return repr(cls())


class Scalar(Describable, metaclass=ScalarMeta):
    def __init__(self, *enum_values):
        self._enum = enum_values
        super().__init__()

    def __repr__(self):
        type_name = self.__class__.type_name
        enum_str, desc_str = "", ""
        if self._enum:
            values = ";".join(self._enum)
            type_name += f"<{values}>"
        if self.desc:
            return f"{type_name} /* {self.desc} */"
        else:
            return type_name

    def __or__(self, other):
        """Make a choice of scalars.
        """
        return Union(self, other)


class String(Scalar):
    type_name = "string"


class Number(Scalar):
    type_name = "number"


def complex():
    param = {
        "v": Number // "Chemical name of the component",
        "u": String // "Units",
        "i": (Number | String) // "Index"
    }

    component = {
        "component": {
            "name": String // "Component name",
            "elements": [String // "Name of an individual element"],
            "valid_phase_types": [String("Liq", "Vap")],
            "type": String("solvent", "solute") // "Type of thing",
            "phase_equilibrium_form": {"Vap": String, "Liq": String},
            "parameter_data": {
                "mw": param,
                "pressure_crit": param,
                "/.*_coeff/": param,
                "/.*_ref/": param,
            },
        }
    }
    return param, component


x = {"foo": [String, "this is a foo"]}


if __name__ == "__main__":
    pprint(x)
    param, component = complex()
    pprint(component)