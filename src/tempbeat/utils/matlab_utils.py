# Utils to get matlab engine


def set_matlab():
    try:
        import matlab.engine
    except ImportError:
        raise ImportError("Matlab engine is not installed.")
    global global_eng
    global_eng = matlab.engine.start_matlab()


def get_matlab():
    global global_eng
    eng = global_eng
    return eng


def quit_matlab():
    global global_eng
    global_eng.quit()
