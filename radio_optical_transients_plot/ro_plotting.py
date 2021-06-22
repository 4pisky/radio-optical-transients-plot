"""Radio Optical plotting functions.

This script contains plotting functions used by the RadioOptical plotting
package.

This file can be imported as a module and contains the following
functions:

    * plotlines - Plots the distance background lines.
    * plotboxpoints - Gets the plotting points for the QSO box.
    * addtracklabels - Adds labels to a dynamic track.
    * extinction_arrow - Draws the extinction arrow on an axes.
"""

import numpy as np

from collections import Counter
from matplotlib.pyplot import Axes
from matplotlib.patches import Ellipse
from radio_optical_transients_plot.ro_utils import OpticaltomJy
from typing import Tuple, List


def plotlines(
    optical_start: List[float], ratio_wanted: List[float]
) -> Tuple[List[float], List[float], List[float]]:
    """Draws the 'Distance Lines' for the main plot.

    Takes inputs of desired optical and radio limits of where the distance
    lines should be drawn on the plot.

    Args:
        optical_start: The optical flux values where the lines should start
            from.
        radio_wanted: The desired radio fluxes where the lines should end.

    Return:
        The optical points to plot along with the radio range to plot.
        Also returns the radio wanted (I can't remember why!).
    """
    radio_range = []
    radio_range.reverse()
    optical_points = []
    OFLUX = optical_start
    optical_points.append(OFLUX)
    this_radio = optical_points[0] * ratio_wanted
    radio_range.append(this_radio)

    while this_radio < 99999.:
        this_radio *= 10.
        NEWOFLUX = this_radio / ratio_wanted
        optical_points.append(NEWOFLUX)
        radio_range.append(this_radio)

    return optical_points, radio_range, ratio_wanted


def cleanlines(
    x: List[float], y: List[float]
) -> Tuple[List[float], List[float]]:
    """Clean the distance lines.

    (I think) this function cleans the distance lines such that they don't
    extend too far.

    Args:
        x: The x coordinates of a line.
        y: The y coordinates of a line.

    Returns:
        The cleaned x and y coordinates.
    """
    clean_x = []
    clean_y = []
    min_x = 1e-6
    max_x = 1e8
    min_y = 5e-4
    max_y = 1e6
    for i, j in zip(x, y):
        if min_x <= i <= max_x:
            if min_y <= j <= max_y:
                clean_x.append(i)
                clean_y.append(j)
    return clean_x, clean_y


def plotboxpoints(
    x1: float, x2: float, y1: float, y2: float
) -> Tuple[float, float]:
    """Obtains the plotting points for the quasar extension box.

    I must admit I can't remember exactly what points this obtains.
    x4 is not used in ro_plot. I think it's something to do with the 'cut'
    in the polygon.

    Args:
        x1: x coordinate 1 of the polygon.
        x2: x coordinate 2 of the polygon.
        y1: y coordinate 1 of the polygon.
        y2: y coordinate 2 of the polygon.

    Returns:
        The coordinates for x3 and x4 (?).

    """
    ratio = y1 / x1
    x3 = y2 / ratio
    ratio = y1 / x2
    x4 = y2 / ratio

    return x3, x4


def addtracklabels(
    x: np.ndarray,
    y: np.ndarray,
    axes: Axes,
    ts: np.ndarray,
    c: str,
    plotall: bool,
    thislabel: str,
    cvs: List[str],
    seonly: bool,
    count: Counter,
    name: str,
    zorder: int = 16
) -> None:
    """Adds labels to the track lines. Specifically the start and end labels
    and the day numbers.

    Plots directly the axes provided.

    Args:
        x: x coordinates of the track (optical).
        y: y coordinates of the track (radio).
        axes: The plot axes.
        ts: Array containing the day number of the track data points.
        c: The type string.
        plotall: True if the summary style is being used.
        thislabel: The label to use for the track.
        cvs: List of tracks that are CVs.
        seonly: True if 'start_end_only' is being used.
        count: Counter object for the plots.
        name: Name of the object.
        zorder: zorder to use.

    Returns:
        None.
    """
    default_shift_x = -0.1
    default_shift_y = -0.09
    nums = {"GRB": 3, "SN": 4}
    quasarlabels = ["BL Lacertae", "BL Lacertae (3 month)", "3C 454.3"]
    sn_labelshifts = {
        "E10": [-0.4, default_shift_y],
        "S14": [default_shift_x, -0.01],
        "E1": [default_shift_x, -0.25],
        "E13": [-0.4, -0.2],
        "S12": [-0.15, -0.2],
        "E6": [-0.25, default_shift_y],
        "E14": [-0.4, -0.19],
        "E9": [-0.25, default_shift_y],
        "S1": [default_shift_x, -0.2],
        "E4": [default_shift_x, -0.2]
    }
    xrb_labelshifts = {"E4": [-0.3, -0.15], "E5": [-0.25, default_shift_y]}

    if not plotall:
        axes.plot(
            x[0],
            y[0],
            marker="o",
            zorder=zorder,
            fillstyle='none',
            color='#32cd32',
            markersize=40,
            mew=2.5
        )
        axes.plot(
            x[-1],
            y[-1],
            marker="o",
            zorder=zorder,
            fillstyle='none',
            color='#dc143c',
            markersize=40,
            mew=2.5
        )
    else:
        if seonly:
            if thislabel == "XRB":
                tag = "S{}".format(count)
                if tag in xrb_labelshifts:
                    xs_shift = xrb_labelshifts[tag][0]
                    ys_shift = xrb_labelshifts[tag][1]
                else:
                    xs_shift = default_shift_x
                    ys_shift = default_shift_y
                endtag = "E{}".format(count)
                if endtag in xrb_labelshifts:
                    xe_shift = xrb_labelshifts[endtag][0]
                    ye_shift = xrb_labelshifts[endtag][1]
                else:
                    xe_shift = default_shift_x
                    ye_shift = default_shift_y
            elif thislabel == "SN":
                tag = "S{}".format(count)
                if tag in sn_labelshifts:
                    xs_shift = sn_labelshifts[tag][0]
                    ys_shift = sn_labelshifts[tag][1]
                else:
                    xs_shift = default_shift_x
                    ys_shift = default_shift_y
                endtag = "E{}".format(count)
                if endtag in sn_labelshifts:
                    xe_shift = sn_labelshifts[endtag][0]
                    ye_shift = sn_labelshifts[endtag][1]
                else:
                    xe_shift = default_shift_x
                    ye_shift = default_shift_y
            else:
                xs_shift = default_shift_x
                ys_shift = default_shift_y
                xe_shift = default_shift_x
                ye_shift = default_shift_y

            axes.text(
                x[0] + (x[0] * xs_shift),
                y[0] + (y[0] * ys_shift),
                "S{}".format(count),
                zorder=zorder,
                weight='bold',
                size=14,
                color='g'
            )
            axes.text(
                x[-1] + (x[-1] * xe_shift),
                y[-1] + (y[-1] * ye_shift),
                "E{}".format(count),
                zorder=zorder,
                weight='bold',
                size=14,
                color='r'
            )
        else:
            axes.text(
                x[0] - (x[0] * 0.09),
                y[0] - (y[0] * 0.09),
                "s",
                zorder=zorder,
                weight='bold',
                size=13
            )
            axes.text(
                x[-1] - (x[-1] * 0.09),
                y[-1] - (y[-1] * 0.09),
                "e",
                zorder=zorder,
                weight='bold',
                size=13
            )

    if not plotall:
        if len(ts) > 0:
            if c == "SN" or c == "GRB":
                axes.text(
                    x[0] - (x[0] * 0.09),
                    y[0] - (y[0] * 0.09),
                    "{}".format(int(ts[0])),
                    zorder=zorder,
                    weight='bold',
                    size=18
                )
                axes.text(x[-1] - (x[-1] * 0.09),
                          y[-1] - (y[-1] * 0.09),
                          "{}".format(int(ts[-1])),
                          zorder=zorder,
                          weight='bold',
                          size=18)
            else:
                if thislabel != "Quasar":
                    alt_ts = [i - ts[0] for i in ts]
                    ts = alt_ts
                    axes.text(x[0] - (x[0] * 0.09),
                              y[0] - (y[0] * 0.09),
                              "{}".format(int(ts[0])),
                              zorder=zorder,
                              weight='bold',
                              size=18)
                    axes.text(x[-1] - (x[-1] * 0.09),
                              y[-1] - (y[-1] * 0.09),
                              "{}".format(int(ts[-1])),
                              zorder=zorder,
                              weight='bold',
                              size=18)
            if c == "SN" or c == "GRB":
                if name == "GRB 100418A":
                    num = 3
                    start = 2
                elif name == "GRB 970508":
                    num = 3
                    start = 2
                else:
                    num = nums[c]
                    start = num
            elif thislabel in cvs:
                num = 2
                start = 1
                # print "Yes"
            elif name == "XTE J1859+226":
                num = 10
                start = 2
            elif name == "GX 339-4":
                num = 3
                start = 2
            else:
                num = 3
                start = 3
            for i in range(start, len(x) - 2, num):
                if name == "GRB 970508" and int(ts[i]) == 50:
                    axes.text(x[i] - (x[i] * 0.15),
                              y[i] - (y[i] * 0.08),
                              "{}".format(int(ts[i])),
                              zorder=zorder,
                              weight='bold',
                              size=18)
                else:
                    axes.text(x[i] - (x[i] * 0.1),
                              y[i] - (y[i] * 0.08),
                              "{}".format(int(ts[i])),
                              zorder=zorder,
                              weight='bold',
                              size=18)


def extinction_arrow(
    a: Axes,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    bg: bool,
    group_agn: bool = False
) -> None:
    """Draws the extinction arrow onto the axes.

    Args:
        a: The axes on which to plot the arrow.
        x1: The start x coordinate of the arrow.
        y1: The start y coordinate of the arrow.
        x2: The end x coordinate of the arrow.
        y2: The end y coordinate of the arrow.
        bg: Whether the background mode should be used or not.
        group_agn: True or False depending on whether the quasars are
            grouped.

    Returns:
        None.
    """
    if bg:
        if group_agn:
            thisc = 'k'
        else:
            thisc = 'darkgray'
    else:
        thisc = 'k'
    opt5 = {
        'head_width': 6. * 1e-3,
        'head_length': 3. * 1.e2,
        'width': 2 * 1.e-4,
        'length_includes_head': True,
        'color': thisc,
        'shape': 'left',
        'zorder': 20
    }
    a.arrow(x1, y1, x2, y2, **opt5)
    a.text(
        OpticaltomJy(8, "R"),
        y1 + (y1 * 0.2),
        r"$\mathbf{A_{R}\ =\ 5\ mag}$",
        color=thisc,
        weight='bold',
        size=20,
        zorder=22
    )
