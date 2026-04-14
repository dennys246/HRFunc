"""
Internal utilities shared between hrfunc.py and hrtree.py.

Lives in its own module to break the circular import that used to exist
between hrfunc.py (imports hrtree) and hrtree.py (imported hrfunc back
for standardize_name, _is_oxygenated, and __file__).

Nothing in this module should import from hrfunc.py or hrtree.py.
"""

import os
import re


_LIB_DIR = os.path.dirname(os.path.abspath(__file__))


def standardize_name(ch_name):
    """
    Standardize channel names to a common format for processing.

    Arguments:
        ch_name (str) - Original channel name

    Returns:
        ch_name (str) - Standardized channel name

    Raises:
        TypeError - If ch_name is not a string
        ValueError - If ch_name is too short to carry an oxygenation suffix
    """
    if not isinstance(ch_name, str):
        raise TypeError(
            f"standardize_name expected a str, got {type(ch_name).__name__}"
        )
    if len(ch_name) < 3:
        raise ValueError(
            f"Channel name {ch_name!r} is too short to standardize; "
            "expected at least 3 characters carrying an oxygenation suffix "
            "(e.g. 'hbo', 'hbr', or wavelength digits)"
        )
    ch_name = re.sub(r'[_\-\s]+', '_', ch_name.lower())
    oxygenation = _is_oxygenated(ch_name)
    if oxygenation:
        ch_name = ch_name[:-3] + 'hbo'
    else:
        ch_name = ch_name[:-3] + 'hbr'
    return ch_name


def _is_oxygenated(ch_name):
    """
    Check whether the channel is HbR or HbO.

    Arguments:
        ch_name (str) - Channel name to check

    Returns:
        bool - True if oxygenated, False if deoxygenated

    Raises:
        ValueError - If channel name has no recognizable oxygenation pattern
        LookupError - If wavelength digits are present but out of range
    """
    if ch_name[-2] == 'b':
        split = ch_name.split('hb')
        if split[1][0] == 'o':
            return True
        elif split[1][0] == 'r':
            return False
        else:
            raise ValueError(
                f"Channel {ch_name} oxygenation status could not be determined, "
                "ensure each channel has appropriate naming scheme with HbO/HbR included"
            )
    elif ch_name[-1] == '0':
        try:
            wavelength = int(ch_name[-3:])
        except (ValueError, TypeError):
            raise LookupError(f"Failed to evaluate oxygenation status of channel {ch_name}")
        if 760 <= wavelength <= 780:
            return False
        elif 830 <= wavelength <= 850:
            return True
        else:
            raise LookupError(
                f"Wavelength found, but failed to evaluate oxygenation status of channel {ch_name}"
            )
    else:
        raise ValueError(
            f"Could not determine oxygenation for channel {ch_name}: "
            "no hb suffix or recognized wavelength pattern"
        )