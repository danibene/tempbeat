from typing import Any, Optional

global_eng: Optional[Any] = None  # Use Any for the type hint


def set_matlab() -> None:
    global global_eng
    try:
        import matlab.engine

        global_eng = matlab.engine.start_matlab()
    except ImportError:
        raise ImportError("Matlab engine is not installed.")


def get_matlab() -> Any:
    global global_eng
    if global_eng is None:
        raise Exception("Matlab engine is not initialized. Call set_matlab() first.")
    return global_eng


def quit_matlab() -> None:
    global global_eng
    if global_eng is not None:
        global_eng.quit()
        global_eng = None  # Properly handle reinitialization cases
