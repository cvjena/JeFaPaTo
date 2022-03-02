__all__ = ["start_up_tasks", "StartUpState"]

import importlib
import importlib.util
import inspect
import pathlib
import typing

from qtpy import QtWidgets

from frontend.start_up.enum_state import StartUpState


class StartUpException(Exception):
    pass


start_up_tasks: typing.List[callable] = []

for files in pathlib.Path(__file__).parent.glob("start_up_*.py"):
    # this feels really hacky, but it works
    # inspired by the way pytest loads tests

    module = files.with_suffix("").relative_to(
        pathlib.Path(__file__).parent.parent.parent
    )
    module = ".".join(module.parts)

    spec = importlib.util.spec_from_file_location(module, files)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    functions = inspect.getmembers(mod, inspect.isfunction)

    for f_name, f_value in functions:
        if f_name.startswith("start_up_"):
            sig = inspect.signature(f_value)
            if "splash_screen" not in sig.parameters:
                raise StartUpException(
                    f"{f_name} must have a parameter named 'splash_screen'"
                )
            if sig.parameters["splash_screen"].annotation != QtWidgets.QSplashScreen:
                raise StartUpException(
                    f"{f_name} must have a parameter named 'splash_screen'"
                    " of type QtWidgets.QSplashScreen"
                )

            if typing.get_type_hints(f_value).get("return") != StartUpState:
                raise StartUpException(
                    f"{f_name} must have a return type of StartUpState"
                )

            start_up_tasks.append(f_value)
