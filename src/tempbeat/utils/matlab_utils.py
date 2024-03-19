# Utils to get matlab engine
from typing import Any


def set_matlab() -> None:
    try:
        import matlab.engine
    except ImportError:
        raise ImportError("Matlab engine is not installed.")
    global global_eng  # type: ignore
    global_eng = matlab.engine.start_matlab()


def get_matlab() -> Any:
    global global_eng  # type: ignore
    eng = global_eng
    return eng


def quit_matlab() -> None:
    global global_eng
    global_eng.quit()
