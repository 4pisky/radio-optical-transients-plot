#!/usr/bin/env python
"""
A script to generate standard radio optical plots as found in Stewart
et al 2018.

Example:
   `radio_optical_figures -h`
"""
import argparse
import matplotlib.pyplot as plt

from radio_optical_transients_plot.ro_main import (
    RadioOpticalPlot, RadioOpticalTrackPlot
)


def parse_args() -> argparse.Namespace:
    """
    Parse the arguments.

    Returns:
        The argument namespace.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--figure",
        type=str,
        choices=[
            '1', '2', '3', '4', '5a', '5b',
            '6a', '6b', '6c', '6d', '7', '8a', '8b', '9',
            'a1a', 'a1b', 'a2', 'a3', 'a4'
        ],
        required=True,
        help="Specify what figure to produce."
    )

    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        default=False,
        help=(
            "Use this option to automatically save the figures "
            "instead of displaying."
        )
    )

    parser.add_argument(
        "-f",
        "--save-format",
        default='png',
        choices=['png', 'pdf'],
        help=(
            "Select the format the figure will be saved as."
        )
    )


    args = parser.parse_args()

    return args


def main() -> None:
    """The main function.

    Returns:
        None
    """
    # numexpr.set_num_threads(2)
    args = parse_args()

    f_num = args.figure

    if f_num == "1":
        plot = RadioOpticalPlot(group_stellar=True)
        fig = plot.generate_plot()

    if f_num == "2":
        plot = RadioOpticalPlot(group_stellar=True)
        fig = plot.generate_plot(
            meerkat=True, show_complete=True, square=True,
            hide_line_labels=True
        )

    if f_num == "3":
        plot = RadioOpticalPlot(group_stellar=True, group_agn=True)
        fig = plot.generate_plot(
            square=True, hide_line_labels=True, schematic_cover=True
        )

    if f_num == "4":
        plot = RadioOpticalPlot(group_agn=True)
        fig = plot.generate_plot(
            square=True, hide_line_labels=True,
            background=True, hide_arrow=True, zoom_plot="0.5,9e-3,2e9,1e3",
            color_list=(
                "Stellar: Star,Stellar: RS CVn,Stellar: Algol,"
                "Stellar: Variable Star,Stellar: "
                "Symbiotic,Stellar: YSO,Stellar: Other,CV"
            ),
        )

    if f_num == "5a":
        plot = RadioOpticalTrackPlot(group_agn=True, group_stellar=True)
        fig = plot.generate_track_plot(
            group_tracks=True, summary_style=True, hide_line_labels=True,
            background=True, zoom_plot="1.43977e-5,0.00580767,111849,27519.4"
        )

    if f_num == "5b":
        plot = RadioOpticalTrackPlot(group_agn=True, group_stellar=True)
        fig = plot.generate_track_plot(
            group_tracks=True, summary_style=True, hide_line_labels=True,
            background=True, start_end_only=True,
            zoom_plot="1.43977e-5,0.00580767,111849,27519.4"
        )

    if f_num == "6a":
        plot = RadioOpticalTrackPlot(group_agn=True, group_stellar=True)
        fig = plot.generate_track_plot(
            hide_line_labels=True,
            background=True,
            zoom_plot="0.00653281,0.161976,6671.47,5035.9",
            only_types=["XRB"],
            hide_arrow=True,
            hide_main_legend=True,
            legend_size=15,
            square=True
        )

        plot.add_text("XRBs", 1000, 2000, fontsize=30, weight='bold')

    if f_num == "6b":
        plot = RadioOpticalTrackPlot(group_agn=True, group_stellar=True)
        fig = plot.generate_track_plot(
            hide_line_labels=True,
            background=True,
            zoom_plot="1.74873,0.0457012,31154.5,108.112",
            only_types=["CV"],
            hide_arrow=True,
            hide_main_legend=True,
            legend_size=15,
            square=True
        )

        plot.add_text("CVs", 2.5, 0.06, fontsize=30, weight='bold')

    if f_num == "6c":
        plot = RadioOpticalTrackPlot(group_agn=True, group_stellar=True)
        fig = plot.generate_track_plot(
            hide_line_labels=True,
            background=True,
            zoom_plot="0.00434,0.0710829,540.484,1028.08",
            only_types=["SN"],
            hide_arrow=True,
            hide_main_legend=True,
            legend_size=15,
            square=True
        )

        plot.add_text("SNe", 100, 300, fontsize=30, weight='bold')

    if f_num == "6d":
        plot = RadioOpticalTrackPlot(group_agn=True, group_stellar=True)
        fig = plot.generate_track_plot(
            hide_line_labels=True,
            background=True,
            zoom_plot="1.52e-5,0.009319,17.2954,63.4996",
            only_types=["GRB"],
            hide_arrow=True,
            hide_main_legend=True,
            legend_size=15,
            square=True
        )

        plot.add_text("GRBs", 0.00003, 0.013, fontsize=30, weight='bold')

    if f_num == "7":
        plot = RadioOpticalPlot(group_stellar=True, group_agn=True)
        fig = plot.ratio_histogram()

    if f_num == "8a":
        plot = RadioOpticalPlot(
            group_stellar=True, group_agn=True,
            transients_file="Stripe82_QSOs.txt"
        )
        fig = plot.generate_plot(
            background=True,
            color_list="Quasar,GRB",
            zoom_plot="5.4363e-5,0.0183888,362.768,5511.11",
            hide_arrow=True,
            hide_line_labels=True,
            square=True
        )

    if f_num == "8b":
        plot = RadioOpticalPlot(
            group_stellar=True, group_agn=True
        )
        fig = plot.generate_plot(
            background=True,
            highlight_list="Quasar,Stellar",
            hide_arrow=True,
            hide_line_labels=True,
            push_agn=True,
            push_stellar_dist=1000,
            square=True
        )

    if f_num == "9":
        plot = RadioOpticalPlot(
            group_stellar=True, group_agn=True,
            transients_file="transient_master_table_04072013.txt"
        )
        fig = plot.generate_plot(
            background=True,
            hide_diag_line_labels=True,
            exclude_type=(
                'FIRST Quasar (SDSS Star),FIRST Quasar (SDSS Gal.),'
                'PSR J1012+5307 (Variable)'
            )
        )

    if f_num == "a1a":
        plot = RadioOpticalPlot(group_stellar=True)
        fig = plot.frequency_histogram()

    if f_num == "a1b":
        plot = RadioOpticalPlot(group_stellar=True)
        fig = plot.band_histogram()

    if f_num == "a2":
        plot = RadioOpticalPlot()
        fig = plot.qso_z_histogram()

    if f_num == "a3":
        plot = RadioOpticalPlot(group_stellar=True)
        fig = plot.stellar_distance_histogram()

    if f_num == "a4":
        plot = RadioOpticalPlot(group_stellar=True)
        fig = plot.grb_z_histogram()

    if args.save:
        save_name = f'ro_figure_{f_num}.{args.save_format}'
        fig.savefig(
            save_name, bbox_inches='tight'
        )
        print(f'Saved {save_name}.')
    else:
        plt.show()


if __name__ == '__main__':
    main()
