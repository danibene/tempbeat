"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = tempbeat.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This file can be renamed depending on your needs or safely removed if not needed.

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import argparse
import logging
import sys
import numpy as np
import scipy.io

from tempbeat import __version__
from tempbeat.extraction.heartbeat_extraction import hb_extract
from tempbeat.utils.timestamp import sampling_rate_to_sig_time

__author__ = "danibene"
__copyright__ = "danibene"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from tempbeat.skeleton import fib`,
# when using this Python module as a library.

def extract_peak_times_from_wav(wav_file: str, method: str = "temp") -> np.ndarray:
    """
    Extract peak times from a WAV file.

    This function extracts the peak times from a WAV file using the specified method.

    Parameters
    ----------
    wav_file : str
        The path to the WAV file.
    method : str, optional
        The method used to extract the peak times. Default is "temp".

    Returns
    -------
    np.ndarray
        Array of timestamps corresponding to the peak times.
    """
    sampling_rate, sig = scipy.io.wavfile.read(wav_file)
    sig_time = sampling_rate_to_sig_time(sig, sampling_rate)
    peak_time = hb_extract(sig, sig_time=sig_time, sampling_rate=sampling_rate, method=method)
    # export to csv
    csv_file = str(wav_file).replace(".wav", "_peak_time.csv")
    np.savetxt(csv_file, peak_time, delimiter=",")
    print(f"Peak times saved to {csv_file}")

# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Just a demonstration")
    parser.add_argument(
        "--version",
        action="version",
        version=f"tempbeat {__version__}",
    )
    parser.add_argument(dest="wav_file", help="The path to the WAV file.", type=str)
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    extract_peak_times_from_wav(args.wav_file)
    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m tempbeat.skeleton "test.wav"
    #
    run()
