Changelog
=========

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

Unreleased
------------

Please see all `Unreleased Changes`_ for more information.

0.0.3 - 2024-08-01
--------------------

Fixed
~~~~~

- Hyperlink in README


0.0.2 - 2024-08-01
--------------------

Added
~~~~~

- Add optional percentage parameter in calculation of MAE
- Add multiple parameters and plots of median template and correlation to bandpass_comparison study

Removed
~~~~~

- Remove parts of mod_fixpeaks that are not currently used in the template algorithm


0.0.1 - 2024-03-19
--------------------

Added
~~~~~

- Add modified version of ``fix_peaks`` (adapted from neurokit2) and misc utils functions, from `in-ear-stress repository`_.
- Add template-based heartbeat extraction and various utility functions, from `in-ear-stress repository`_.
- Add function to run a heartbeat extraction method implemented in MATLAB (adapted from code on private GitLab repository).

.. _in-ear-stress repository: https://github.com/danibene/in-ear-stress/commit/2b0679793c9baf05e621e3900d1fa92225a63073
.. _Unreleased Changes: https://github.com/danibene/tempbeat/compare/v0.0.1...HEAD
.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html
