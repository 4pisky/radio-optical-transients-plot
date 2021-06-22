"""Radio Optical plot main classes.

This script contains the main plotting classes for the radio optical plots.

This file can be imported as a module and contains the following
classes:

    * plotlines - Plots the distance background lines.
    * plotboxpoints - Gets the plotting points for the QSO box.
    * addtracklabels - Adds labels to a dynamic track.
    * extinction_arrow - Draws the extinction arrow on an axes.
"""

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.markers as plt_markers
import numpy as np
import pandas as pd
import pkg_resources
import warnings

from astropy.cosmology import FlatLambdaCDM
from collections import Counter
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Ellipse, Arc
from matplotlib.transforms import ScaledTranslation
from pathlib import Path
from pprint import pprint
from typing import List, Optional, Tuple
from radio_optical_transients_plot.ro_utils import (
    ConvertToABMag_pd,
    OpticaltomJy_pd,
    OpticaltomJy,
    mJytoOptical,
    kcorrect,
    ConvertToABMag,
    stellar_dist
)
from radio_optical_transients_plot.ro_plotting import (
    plotlines,
    cleanlines,
    extinction_arrow,
    plotboxpoints,
    addtracklabels
)


def docstring_inherit(parent):
    """Function to pass attributes to inherited class.

    Args:
        parent (class): The parent class.

    Returns:
        obj: The inherit object.
    """
    def inherit(obj):
        """Sorts and inherits the attributes section.

        Args:
            obj: The object.

        Returns:
            obj: The object.
        """
        spaces = "    "
        if not str(obj.__doc__).__contains__("Attributes:"):
            obj.__doc__ += "\n" + spaces + "Attributes:\n"
        obj.__doc__ = str(obj.__doc__).rstrip() + "\n"
        to_loop = (
            parent
            .__doc__
            .split("Attributes:\n")[-1]
            .lstrip()
            .split("\n")
        )
        for attribute in to_loop:
            obj.__doc__ += spaces * 2 + str(attribute).lstrip().rstrip() + "\n"

        return obj

    return inherit


class RadioOpticalData(object):
    """The representation of the radio-optical data object as found in
    Stewart et al. 2018.

    Attributes:
        base_data_file (Path): The path of the data file containing the
            master (base) data.
        exclude_stellar_types (List[str]): The types to exclude from the
            stellar types.
        ab_list (List[str]): The names of objects that have the optical
            magnitudes in the AB system. All those not listed are converted
            to AB.
        group_agn (bool): True when the 'group_agn' option is used.
        group_stellar (bool): True when the 'group_stellar' option is used.
        master_df (pd.DataFrame): The dataframe containing the master data.
        basis (List[str]): The types that form the basis plot.
        transients_data_file (Path): The path of the transient data file
            containing the master (base) data.
        transients_data_df (pd.DataFrame): The dataframe containing the
            transient data.
        transients (List[str]): The types that make up the transient objects.
    """
    def __init__(
        self,
        base_data_file: Optional[str] = None,
        extra_exclude_stellar_types: Optional[List[str]] = None,
        extra_ab_list: Optional[List[str]] = None,
        group_agn: bool = False,
        group_stellar: bool = False,
        transients_file: Optional[str] = None
    ) -> None:
        """
        Init function.

        Args:
            base_data_file: The file containing the tab-separated master data.
                If 'None' is entered then the packaged latest master table will
                be used.
            extra_exclude_stellar_types: Extra stellar types to add to the
                stellar exclude list.
            extra_ab_list: Extra names to add to the AB list.
            group_agn: When 'True' the quasars are grouped together under
                the type 'Quasars'.
            group_stellar: When 'True' the stellar sources are all grouped
                under the type 'Stellar'.
            transients_file: Path to the transients file to load. Also accepts
                the names of the packaged transient files
                'transient_master_table_04072013.txt' and 'Stripe82_QSOs.txt'.

        Returns:
            None.
        """
        super(RadioOpticalData, self).__init__()
        # If None grab the packaged master data file.
        if base_data_file is None:
            self.base_data_file = Path(pkg_resources.resource_filename(
                __name__, "./data/Master_Table_27042018.tsv"
            ))
        else:
            self.base_data_file = Path(base_data_file)
            if not self.base_data_file.exists():
                raise IOError(
                    f'The file {self.base_data_file.resolve()} cannot be'
                    ' found!'
                )

        # The next two are hardcoded due to the master data requiring these
        # filters.
        self.exclude_stellar_types = [
            'XRB',
            'Xray source',
            'RadioGalaxy',
            'QSO',
            'X-rayNova',
            "Planetary Nebula"
        ]

        if extra_exclude_stellar_types is not None:
            self.exclude_stellar_types += extra_exclude_stellar_types

        self.ab_list = [
            "GRB 010921",
            "GRB 051221A",
            "GRB 080319B",
            "GRB 081203B",
            "SN 2008D"
        ]

        if extra_ab_list is not None:
            self.ab_list += extra_ab_list

        self.group_agn = group_agn
        self.group_stellar = group_stellar

        self.base_data_df = self.read_data()

        self.master_df = self.base_data_df.copy()

        self.basis = self.base_data_df["Type"].unique().tolist()

        if transients_file is not None:
            self.transients_data_file = Path(transients_file)
            if not self.transients_data_file.exists():
                # Try looking in the data dir
                self.transients_data_file = Path(
                    pkg_resources.resource_filename(
                        __name__, f"./data/{self.transients_data_file.name}"
                    )
                )
                if not self.transients_data_file.exists():
                    raise IOError(
                        f'The file {self.transients_data_file.resolve()}'
                        ' cannot be found in the specified location or in the'
                        ' packaged data files.'
                    )
            if (self.transients_data_file.name ==
                    "transient_master_table_04072013.txt"):
                first_use_I = True
            else:
                first_use_I = False
            self.transients_data_df = self.read_data(
                transients=True, first_use_I=first_use_I
            )
            self.transients = self.transients_data_df["Type"].unique().tolist()

            self.master_df = self.master_df.append(self.transients_data_df)
        else:
            self.transients = []


    def read_data(
        self, transients: bool = False, first_use_I: bool = False
    ) -> pd.DataFrame:
        """Function to read in the master and transient data from their
        respective files.

        Tab-separated files are expected.

        The first_use_I argument is used as a cheat method to load the file
        'transient_master_table_04072013.txt' as this is the only instance
        where I band magnitudes are used.

        Args:
            transients: Set as 'True' when the data file being read is not
                the base data file.
            first_use_I: Set to `True` when reading the file
                'transient_master_table_04072013.txt'.

        Returns:
            The DataFrame containing the data.
        """
        if transients:
            data_file = self.transients_data_file
        else:
            data_file = self.base_data_file
        base_data = pd.read_csv(data_file, comment="#", sep="\t")
        base_data = self.master_table_analysis(
            base_data, transients=transients, first_use_I=first_use_I
        )

        return base_data

    def master_table_analysis(
        self,
        df: pd.DataFrame,
        transients: bool = False,
        first_use_I: bool = False
    ) -> pd.DataFrame:
        """Analyses the data and performs conversions to magnitudes.

        Column names expected are:
            - 'F Radio / mJy'
            - 'V' or 'R' (or 'I')
            - 'Type'
            - 'Name'

        The first_use_I argument is used as a cheat method to load the file
        'transient_master_table_04072013.txt' as this is the only instance
        where I band magnitudes are used.

        Args:
            df: DataFrame containing the raw data.
            transients: Set as 'True' when the data file being read is not
                the base data file.
            first_use_I: Set to `True` when reading the file
                'transient_master_table_04072013.txt'.

        Returns:
            The DataFrame with new columns calculated:
                - 'radio'
                - 'optical_mag_used_band'
                - 'optical_mag_used_value'
                - 'optical_mag_used_value_processed'
                - 'ratio'
                - 'optical_in_mJy'
        """
        # First we get rid of any objects that have no R or V measurement
        # Cheating as the FIRST transients we use I
        if not first_use_I:
            mask = ((df["V"] == 0.) & (df["R"] == 0.))
            df = df[~mask].reset_index(drop=True)
        df = df[df["F Radio / mJy"] != 0.].reset_index(drop=True)
        df["radio"] = df["F Radio / mJy"]
        # The old structure used a large dictionary, but we don't need that
        # with pandas, do some intermediate processing.
        # First get which band we're using for each source, preference of
        # R -> V.
        # Get mask of which ones are V
        if not first_use_I:
            mask = df["R"] == 0.
            df["optical_mag_used_value"] = np.where(mask, df["V"], df["R"])
            df["optical_mag_used_band"] = np.where(mask, "V", "R")
        else:
            df["optical_mag_used_value"] = df["I"]
            df["optical_mag_used_band"] = "I"

        # Next step is to convert the magnitudes to the AB system if needed
        to_convert_mask = ~(
            (df["Type"].str.contains("OPTICAL Sel.")) |
            (df["Name"].isin(self.ab_list)) |
            (df['optical_mag_used_band'] == "I")
        )

        df["optical_mag_used_value_processed"] = df["optical_mag_used_value"]

        converted_values = df.loc[to_convert_mask, [
            "optical_mag_used_value",
            "optical_mag_used_band"
        ]].apply(ConvertToABMag_pd, axis=1)

        df.loc[to_convert_mask, "optical_mag_used_value_processed"] = (
            converted_values
        )

        df["optical_in_mJy"] = df[[
            "optical_mag_used_value_processed",
            "optical_mag_used_band"
        ]].apply(OpticaltomJy_pd, axis=1)

        df["ratio"] = df["radio"] / df["optical_in_mJy"]

        if not transients:
            if self.group_agn:
                agn_mask = (
                    (df["Type"]=="Quasar (OPTICAL Sel.)") |
                    (df["Type"] == "Quasar (RADIO Sel.)")
                )
                df["Type"] = np.where(agn_mask, "Quasar", df["Type"])

            df = self._sort_stellar_sources(
                df, group_stellar = self.group_stellar,
                stellar_exclude_list = self.exclude_stellar_types
            )

        return df

    def _sort_stellar_sources(
        self,
        df: pd.DataFrame,
        group_stellar: bool = True,
        stellar_exclude_list: List[str] = []
    ) -> pd.DataFrame:
        """The stellar sources are grouped together and excluded.

        Args:
            df: The DataFrame containing the data.
            group_stellar: Indicates whether stellar sources should be grouped
                together.
            stellar_exclude_list: Types of sources to exclude from the stellar
                sources.

        Returns:
            The DataFrame with the stellar sources grouped and excluded.
        """
        new_main_types = {
            "CV": "CV",
            "Magnetic CV": "CV",
            "X": "X-ray binary",
            "RSCVn": "Stellar: RS CVn",
            "Algol": "Stellar: Algol",
            "SymbioticStar": 'Stellar: Symbiotic',
            "VariableStar": 'Stellar: Variable Star',
            "T Tauri": "Stellar: YSO",
            "YSO": "Stellar: YSO",
            "P": "Stellar: PMS",
            "Star": "Stellar: Star"
        }

        stellar_rows = df[df["Type"] == "Stellar"]

        rows_to_drop = stellar_rows[
            stellar_rows["Subtype"].isin(stellar_exclude_list)
        ]

        df = df.drop(rows_to_drop.index).reset_index(drop=True)

        stellar_rows = df[df["Type"] == "Stellar"]

        #By group stellar I mean have them all as one category 'Stellar'
        if not group_stellar:
            # TODO: Remove loop
            for i, row in stellar_rows.iterrows():
                if row["Subtype"] in new_main_types:
                    df.at[i, "Type"] = new_main_types[row["Subtype"]]
                else:
                    df.at[i, "Type"] = "Stellar: Other"

        return df

    def calc_averages(self) -> pd.DataFrame:
        """Calculates the averages of the data.

        Returns:
            DataFrame containing the mean and median information.
        """
        averages = self.base_data_df.groupby("Type").agg({
            'radio': ['mean', 'median'],
            'optical_in_mJy': ['mean', 'median']
        })

        averages['optical_in_Mag', 'mean'] = (
            averages['optical_in_mJy', 'mean'].apply(mJytoOptical, args=("R"))
        )
        averages['optical_in_Mag', 'median'] = (
            averages['optical_in_mJy', 'median'].apply(
                mJytoOptical, args=("R")
            )
        )

        return averages

    def _load_stellar_distances(self) -> pd.DataFrame:
        """Loads the stellar distances data.

        Returns:
            The DataFrame containing the stellar distances information.
            It contains the columns 'object' and 'distance'. 'Object' is
            related to 'Name' in the main data.
        """
        stellar_distance_file = Path(pkg_resources.resource_filename(
            __name__, "./data/stellar_distances.csv"
        ))
        s_dist = pd.read_csv(stellar_distance_file)

        return s_dist

    def _load_qso_redshifts(self) -> pd.DataFrame:
        """Loads the quasar redshift data.

        Returns:
            The DataFrame containing the quasar redshift information.
            It contains the columns 'name' and 'z'.
        """
        qso_redshift_file = Path(pkg_resources.resource_filename(
            __name__, "./data/all_qso_redshifts.txt"
        ))
        redshifts = pd.read_csv(
            qso_redshift_file,
            comment='#',
            sep=";",
            names=['ra', 'dec', 'name', 'z'],
            usecols=['name', 'z'],
            dtype={'name':str, 'z':str},
            na_values=['',]
        )
        redshifts['z'] = redshifts['z'].str.strip()
        redshifts = redshifts.loc[redshifts['z'] != '']
        redshifts['z'] = redshifts['z'].astype(float)

        return redshifts


@docstring_inherit(RadioOpticalData)
class RadioOpticalPlot(RadioOpticalData):
    """The main plotting class for the Radio Optical plot.

    Attributes:
        markers (Dict[str, str]): What matplotlib markers to use for each
            class.
        mark_colors (Dict[str, str]): What matplotlib colors to use for each
            class.
        current_fig (plt.figure): The current figure that has been
            generated.
        hide_legend (bool): Whether the main legend should be hidden.
    """
    def __init__(
        self,
        base_data_file: Optional[str] = None,
        extra_exclude_stellar_types: Optional[List[str]] = None,
        extra_ab_list: Optional[List[str]] = None,
        group_agn: bool = False,
        group_stellar: bool = False,
        transients_file: Optional[str] = None
    ) -> None:
        """Init function.

        Args:
            base_data_file: The file containing the tab-separated master data.
                If 'None' is entered then the packaged latest master table
                will be used.
            extra_exclude_stellar_types: Extra stellar types to add to the
                stellar exclude list.
            extra_ab_list: Extra names to add to the AB list.
            group_agn: When 'True' the quasars are grouped together under
                the type 'Quasars'.
            group_stellar: When 'True' the stellar sources are all grouped
                under the type 'Stellar'.
            transients_file: Path to the transients file to load. Also accepts
                the names of the packaged transient files
                'transient_master_table_04072013.txt' and 'Stripe82_QSOs.txt'.

        Returns:
            None
        """
        # Could do kwargs below but clearer for users to see if they are
        # explicitly passed.
        super().__init__(
            base_data_file,
            extra_exclude_stellar_types=extra_exclude_stellar_types,
            extra_ab_list=extra_ab_list,
            group_agn=group_agn,
            group_stellar=group_stellar,
            transients_file=transients_file
        )

        self.markers = {
            "Quasar": '.',
            "Quasar (OPTICAL Sel.)": '.',
            "Quasar (RADIO Sel.)": '.',
            "Quasar (Stripe82)": 'o',
            "FIRST (SDSS Gal.)": '.',
            "FIRST (SDSS Star)": '.',
            "Variable Quasar": '.',
            "Variable Quasar (Galaxy)": '.',
            "GRB": 's',
            "SN": '^',
            "Magnetar": 'd',
            "CV": 'v',
            "RRAT": 'd',
            "Pulsar": 'd',
            "X-ray binary": 'o',
            "Stellar: Other": "D",
            "Stellar": '>',
            "Stellar: Star": '>',
            "Stellar: RS CVn": 's',
            "Stellar: Algol": (4,1,0),
            "Stellar: Symbiotic": 'd',
            "Stellar: YSO": 'o',
            "Stellar: PMS": ">",
            "Stellar: Variable Star": "^",
            "PSR J1012+5307 (Variable)": (4,1,45),
            "SDSS-QSO (Variable)": (4,1,0),
            "Unclassified (Variable)": (4,1,0),
            "Star (Variable)": (4,1,0),
            "SDSS-GAL (Transient)": '*',
            "Unclassified (Transient)": (10,1,0),
            "GRB140907A (Transient)": (4,1,0)
        }

        self.mark_colors={
            "Quasar":'#377eb8',
            "Quasar (OPTICAL Sel.)":'#377eb8',
            "Quasar (RADIO Sel.)":'#e41a1c',
            "Quasar (Stripe82)":"#9acd32",
            "FIRST (SDSS Gal.)":'#9ACD32',
            "FIRST (SDSS Star)":'#4B0082',
            "Variable Quasar":'b',
            "Variable Quasar (Galaxy)":'y',
            "GRB":'#ff7f00',
            "SN":'#984ea3',
            "Magnetar":'#FF4500',
            "CV":'c',
            "RRAT":'#ffff33',
            "Pulsar":'#f781bf',
            "X-ray binary":'#4daf4a',
            "Stellar: Other":'#FFD700',
            "Stellar: RS CVn":'#D2691E',
            "Stellar: Algol": 'm',
            "Stellar":'#DAA520',
            "Stellar: Star":'#DAA520',
            "Stellar: Symbiotic":'#008B8B',
            "Stellar: YSO":'#F08080',
            "Stellar: PMS":'#CD853F',
            "Stellar: Variable Star":"#1e90ff",
            "PSR J1012+5307 (Variable)":'#00FFFF',
            "SDSS-QSO (Variable)":'#377eb8',
            "Unclassified (Variable)":'#e41a1c',
            "Star (Variable)":'#DAA520',
            "SDSS-GAL (Transient)":'#984ea3',
            "Unclassified (Transient)":'#4daf4a',
            "GRB140907A (Transient)":'#FF00FF'
        }

        self.current_fig = None

    def generate_plot(
        self,
        exclude_type: Optional[str] = None,
        exclude_name: Optional[str] = None,
        qso_box: bool = False,
        push_agn: bool = False,
        push_stellar_dist: float = None,
        meerkat: bool = False,
        ska_lsst: bool = False,
        background: bool = False,
        color_list: Optional[str] = None,
        highlight_list: Optional[str] = None,
        schematic: bool = False,
        schematic_cover: bool = False,
        hide_line_labels: bool = False,
        hide_diag_line_labels: bool = False,
        show_complete: bool = False,
        hide_arrow: bool = False,
        square: bool = False,
        hide_main_legend: bool = False,
        zoom_plot: Optional[str] = None,
        legend_size: int = 18,
    ) -> plt.figure:
        """The main function to generate a radio optical plot.

        Args:
            exclude_type: A string of comma separated types to exclude from
                plotting. E.g. 'SN,GRB'.
            exclude_name: A string of comma separated types to exclude from
                plotting. E.g. 'SN 2004dj,GRB 030329'.
            qso_box: Set to 'True' to show a shaded region that represents
                the area quasars will extend to.
            push_agn: Set to 'True' to show how quasars will appear when
                pushed back by sqrt(10)x and 10x.
            push_stellar_dist: Show how the stellar sources will appear when
                pushed back by a provided distance (units is parsec).
            meerkat: Show the MeerKAT and MeerLICHT sensitivity limit lines.
            ska_lsst: Show the SKA and LSST sensitivity limit lines.
            background: Set to 'True' to show the basis plot as a light grey
                color instead of full color. Useful for track plots or
                transient overlays.
            color_list: A string list of types to show in color. Enter as a
                comma separated string, e.g. 'SN,GRB'.
            highlight_list: Used with 'background'. Highlights the types in a
                darker shade of gray. Enter as a  comma separated string,
                e.g. 'SN,GRB'.
            schematic: Show opaque overlay ellipses of type clouds. Note, do
                not use with 'group_agn'.
            schematic_cover: Show transparent overlay ellipses of type clouds.
            hide_line_labels: Set to 'True' to hide the labels on the SDSS and
                FIST survey limit lines.
            hide_diag_line_labels: Set to 'True' to hide the labels on the
                distance lines.
            show_complete: When 'True' only those sources in the 'complete'
                region of the plot are shown in color.
            hide_arrow: Hide the extinction arrow.
            square: Set the plot shape to be square instead of rectangle.
            hide_main_legend: Hide the main legend of the plot.
            zoom_plot: Zoom to area of the plot, enter as 'x1,y1,x2,y2'.
            legend_size: The size of the legend with in square mode.

        Returns:
            The resulting figure.
        """
        if push_agn and not self.group_agn:
            raise ValueError(
                "Data must be initialised with 'group_agn=True"
                " to use the 'push_agn' option!"
            )
        #change distance settings
        if push_stellar_dist is not None:
            change_stellar_distanes = True
        else:
            change_stellar_distanes = False

        #Creates exclude lists
        if exclude_name is not None:
            xname = exclude_name.split(',')
        else:
            xname = []

        if exclude_type is not None:
            xtype = exclude_type.split(',')
        else:
            xtype = []

        # Sort out color list
        if color_list is None:
            color_list = []
        else:
            color_list = color_list.split(",")

        # Sort out highlight list
        if highlight_list is None:
            highlight_list = []
        else:
            highlight_list = highlight_list.split(",")

        # Set the figure size
        if square:
            fig = plt.figure(1, figsize=(16.,11.25))
            title_font_size = 30
            self.prop = fm.FontProperties(size=legend_size)
            label_size = 30
            ticksize = 25
        else:
            fig = plt.figure(1, figsize=(20.,11.25))
            title_font_size = 20
            self.prop = fm.FontProperties(size=14)
            label_size = 'xx-large'
            ticksize = 20

        self.hide_legend = hide_main_legend

        # Create the axis
        ax1 = fig.add_subplot(111)

        #This sets up the top axis showing the magnitude
        ax6 = ax1.twiny()
        ax6.set_xscale('log')
        ax6.set_xlim([0.00001, 10000000])

        # Set the ticks for ax6
        magrange = range(0, 30, 2)
        ticks = [OpticaltomJy(thismag, "R") for thismag in magrange]
        ticklabels = magrange
        ax6.set_xticks(ticks)
        ax6.set_xticklabels(ticklabels)

        # Add title for ax6
        ax1.text(
            0.5, 1.06, 'Optical AB magnitude (mag)',
            horizontalalignment='center',
            fontsize=title_font_size,
            transform = ax1.transAxes
        )

        # Set up the diagonal distance lines
        linesmagrange = range(-25, 34, 2)
        lineticks = [OpticaltomJy(thismag, "R") for thismag in linesmagrange]
        # Manually set the locations
        ratio_wanted=[
            1e9, 1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1, 1e-1,
            1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10
        ]
        optical_range = lineticks
        optical_range.reverse()

        for i in ratio_wanted:
            lineopt, linerad, theratio = plotlines(optical_range[0], i)
            lineopt, linerad = cleanlines(lineopt, linerad)
            if len(lineopt)==0:
                continue
            ax1.plot(
                lineopt, linerad, linestyle='--', color='gray', marker=None,
                linewidth=1.5, zorder=1, alpha=0.6
            )

            if not np.any([hide_line_labels, hide_diag_line_labels]):
                if theratio == 1e7:
                    ax1.text(
                        OpticaltomJy(
                            mJytoOptical(lineopt[-1], "R",) + 3.4, "R"
                        ),
                        linerad[-1] - 2.1e5,
                        "{0:.0e}".format(theratio),
                        rotation=40, alpha=0.6
                    )
                elif 1e3 <= theratio <= 1e9:
                    ax1.text(
                        OpticaltomJy(
                            mJytoOptical(lineopt[-1], "R",) + 2.5, "R"
                        ),
                        linerad[-1] - 1.9e5,
                        "{0:.0e}".format(theratio),
                        rotation=40, alpha=0.6
                    )
                elif theratio == 1:
                    ax1.text(
                        OpticaltomJy(
                            mJytoOptical(lineopt[0], "R",) - 0.8, "R"
                        ),
                        linerad[0] + 0.35e-2,
                        "{0:.0e}".format(theratio),
                        rotation=40, alpha=0.6
                    )
                elif 1e-10 < theratio < 1e3:
                    ax1.text(
                        OpticaltomJy(
                            mJytoOptical(lineopt[0], "R",) - 0.6, "R"
                        ),
                        linerad[0] + 0.4e-2,
                        "{0:.0e}".format(theratio),
                        rotation=40, alpha=0.6
                    )

        # Add the QSO box if requested
        if qso_box:
            SDSSLimit = OpticaltomJy(22.2, "R")
            agnx1 = 0.04# SDSSLimit
            agnx2 = 0.8
            agny1 = 1000.
            agny2 = 0.03
            agny3 = 1.
            agnx3, _ = plotboxpoints(agnx1, agnx2, agny1, agny3)
            ax1.add_patch(
                Polygon([
                    [agnx1, agny1],
                    [agnx2,agny1],
                    [agnx2,agny2],
                    [SDSSLimit,agny2],
                    [agnx3,1.]
                ], closed=True, facecolor='#377eb8', edgecolor='#377eb8',
                label="Likely Quasar Region",
                alpha=0.4, zorder=0)
            )
            ax1.add_patch(
                Polygon([
                    [agnx1, agny1],
                    [agnx2,agny1],
                    [agnx2,agny2],
                    [SDSSLimit,agny2],
                    [agnx3,1.]
                ], closed=True, facecolor='None', edgecolor="#377eb8",
                alpha=0.4, zorder=0, hatch="\\")
            )

        # zorder counter
        num = 3

        combined_list = self.basis + self.transients

        # Now start looping through per type to plot
        for i in combined_list:
            if i in xtype:
                continue

            if i in self.basis:
                if i == "Stellar":
                    size_toplot = 50
                else:
                    size_toplot = 90
            else:
                if "(Variable)" in i:
                    size_toplot = 300
                elif "(Transient)" in i:
                    size_toplot = 600
                else:
                    size_toplot = 100

            # Get the selection
            data_selection = (
                self.master_df.loc[self.master_df["Type"] == i]
            )

            # Filter out not wanted objects.
            if len(xname) > 0:
                data_selection = (
                    data_selection[~data_selection["Name"].isin(xname)]
                )

            try:
                # Obtain marker style and color
                mark_color = self.mark_colors[i]
                marker = self.markers[i]
            except:
                mark_color = None
                marker = '*'


            if i in self.basis:
                # Change some label headings
                if i == 'SN':
                    thislabel = "Supernova"
                elif i == 'Pulsar':
                    thislabel = "Radio Pulsar"
                else:
                    thislabel = i

                if background and i not in color_list:
                    num = 3
                    if i in highlight_list:
                        thiscolor = 'darkgrey'
                    else:
                        thiscolor = 'lightgrey'
                    ax1.scatter(
                        data_selection["optical_in_mJy"],
                        data_selection["radio"],
                        s=size_toplot,
                        color=thiscolor,
                        marker=marker,
                        label=thislabel,
                        zorder=num
                    )

                elif not schematic and not show_complete:
                    ax1.scatter(
                        data_selection["optical_in_mJy"],
                        data_selection["radio"],
                        s=size_toplot,
                        color=mark_color,
                        marker=marker,
                        label=thislabel,
                        zorder=num
                    )

                elif show_complete:
                    color_ones_radio = []
                    color_ones_optical = []
                    bg_ones_radio = []
                    bg_ones_optical = []
                    o_limit = OpticaltomJy(22.2, "R")
                    r_limit = 1.0
                    for p, val in enumerate(data_selection["optical_in_mJy"]):
                        thisradio = data_selection["radio"].iloc[p]
                        if (thisradio > r_limit) and (val > o_limit):
                            color_ones_optical.append(val)
                            color_ones_radio.append(thisradio)
                        else:
                            bg_ones_optical.append(val)
                            bg_ones_radio.append(thisradio)
                    if len(color_ones_radio)>0:
                        ax1.scatter(
                            color_ones_optical,
                            color_ones_radio,
                            s=size_toplot,
                            color=mark_color,
                            marker=marker,
                            label=thislabel,
                            zorder=num
                        )
                        ax1.scatter(
                            bg_ones_optical,
                            bg_ones_radio,
                            s=size_toplot,
                            color='lightgrey',
                            marker=marker,
                            zorder=num
                        )
                    else:
                        ax1.scatter(
                            bg_ones_optical,
                            bg_ones_radio,
                            s=size_toplot,
                            color='lightgrey',
                            marker=marker,
                            label=thislabel,
                            zorder=num
                        )

                if schematic:
                    toplot = [
                        "Quasar (OPTICAL Sel.)", 'Stellar', "GRB",
                        "SN", "Pulsar", "X-ray binary"
                    ]
                    heights = {
                        "Quasar (OPTICAL Sel.)": 4.5, 'Stellar': 3,
                        "GRB": 1.8, "SN": 1.5, "Pulsar": 3,
                        "X-ray binary": 4.5
                    }
                    widths = {
                        "Quasar (OPTICAL Sel.)": 3.2, 'Stellar': 6,
                        "GRB": 3.3, "SN": 2.75, "Pulsar": 1.8,
                        "X-ray binary": 1.25
                    }
                    positions = {
                        "Quasar (OPTICAL Sel.)": (0.2,1), 'Stellar': (0.5,0),
                        "GRB": (-0.3,0), "SN": (0.2,0), "Pulsar": (-0.5,0.2),
                        "X-ray binary": (-0.5,0)
                    }
                    angles = {
                        "Quasar (OPTICAL Sel.)": -20, 'Stellar': 0,
                        "GRB": 10, "SN": 0, "Pulsar": 0, "X-ray binary": -40
                    }

                    textlabels = {
                        "Quasar (OPTICAL Sel.)": "Quasars",
                        'Stellar': "Stellar & CVs",
                        "GRB": "GRBs",
                        "SN": "SNe",
                        "Pulsar": " Radio\nPulsars",
                        "X-ray binary": "XRBs"
                    }

                    if i in toplot:
                        circ_offset = ScaledTranslation(
                            np.median(data_selection["optical_in_mJy"]),
                            np.median(data_selection["radio"]),
                            ax1.transScale
                        )

                        circ_tform = (
                            circ_offset + ax1.transLimits + ax1.transAxes
                        )

                        thisopticalmedian = np.median(
                            data_selection["optical_in_mJy"]
                        )
                        thisradiomedian = np.median(
                            data_selection["radio"]
                        )
                        thisopticalstd = np.std(
                            data_selection["optical_in_mJy"]
                        )
                        thisradiostd = np.std(data_selection["radio"])
                        optmax = max(data_selection["optical_in_mJy"])
                        radmax = max(data_selection["radio"])
                        optmin = min(data_selection["optical_in_mJy"])
                        radmin = min(data_selection["radio"])
                        e1 = Ellipse(
                            xy=positions[i], width=widths[i],
                            height=heights[i], transform=circ_tform,
                            alpha=0.5, zorder=20, angle=angles[i],
                            color=mark_color
                        )
                        ax1.add_artist(e1)

                        if i == "X-ray binary":
                            ax1.text(
                                positions[i][0]-0.3,
                                positions[i][1],
                                textlabels[i],
                                transform=circ_tform,
                                zorder=21,
                                weight='bold',
                                size=24
                            )
                        else:
                            ax1.text(
                                positions[i][0] - 0.7,
                                positions[i][1],
                                textlabels[i],
                                transform=circ_tform,
                                zorder=21,
                                weight='bold',
                                size=24
                            )

                if schematic_cover:
                    toplot = [
                        "Quasar", 'Stellar', "GRB", "SN",
                        "Pulsar", "X-ray binary"
                    ]
                    heights = {
                        "Quasar": 4.9, 'Stellar': 3.8, "GRB": 1.8,
                        "SN": 1.5, "Pulsar": 3, "X-ray binary": 6.5
                    }
                    widths = {
                        "Quasar": 3.2, 'Stellar': 6.5, "GRB": 3.6,
                        "SN": 2.75, "Pulsar": 1.8, "X-ray binary": 2.5
                    }
                    positions = {
                        "Quasar": (0.2,1), 'Stellar': (0.2,0.1),
                        "GRB": (-0.1,0), "SN": (0.2,0), "Pulsar": (-0.5,0.2),
                        "X-ray binary": (0.2,0.3)
                    }
                    angles = {
                        "Quasar": -20, 'Stellar': 0, "GRB": 10, "SN": 0,
                        "Pulsar": 0, "X-ray binary": -45
                    }
                    textlabels={
                        "Quasar": "Quasars", 'Stellar': "Stellar & CVs",
                        "GRB": "GRBs", "SN": "SNe",
                        "Pulsar": " Radio\nPulsars",
                        "X-ray binary": "XRBs"
                    }
                    if i in toplot:
                        circ_offset = ScaledTranslation(
                            np.median(data_selection["optical_in_mJy"]),
                            np.median(data_selection["radio"]),
                            ax1.transScale
                        )

                        circ_tform = (
                            circ_offset + ax1.transLimits + ax1.transAxes
                        )

                        thisopticalmedian = np.median(
                            data_selection["optical_in_mJy"]
                        )
                        thisradiomedian = np.median(data_selection["radio"])
                        thisopticalstd = np.std(
                            data_selection["optical_in_mJy"]
                        )
                        thisradiostd = np.std(data_selection["radio"])
                        optmax = max(data_selection["optical_in_mJy"])
                        radmax = max(data_selection["radio"])
                        optmin = min(data_selection["optical_in_mJy"])
                        radmin = min(data_selection["radio"])

                        if i=="Quasar":

                            e1 = Arc(
                                (0.2,1), height=4.9, width=3.4,
                                transform=circ_tform, zorder=20,
                                theta1=324.9, theta2=244,
                                hatch = '..........', angle=-20, alpha=0.2
                            )
                            e1.set_color(mark_color)

                        else:
                            e1 = Ellipse(
                                xy=positions[i], width=widths[i],
                                height=heights[i], transform=circ_tform,
                                alpha=0.2, zorder=50, angle=angles[i],
                                color=mark_color
                            )
                        ax1.add_artist(e1)

                        if i == "X-ray binary":
                            ax1.text(
                                positions[i][0] - 0.3,
                                positions[i][1],
                                textlabels[i],
                                transform=circ_tform,
                                zorder=21,
                                weight='bold',
                                size=24
                            )
                        elif i == "SN":
                            ax1.text(
                                positions[i][0] - 0.4,
                                positions[i][1] - 0.2,
                                textlabels[i],
                                transform=circ_tform,
                                zorder=21,
                                weight='bold',
                                size=24
                            )
                        else:
                            ax1.text(
                                positions[i][0] - 0.7,
                                positions[i][1],
                                textlabels[i],
                                transform=circ_tform,
                                zorder=21,
                                weight='bold',
                                size=24
                            )
            else:
                if i == "FIRST (SDSS Gal.)" or i=="FIRST (SDSS Star)":
                    ax1.scatter(
                        data_selection["optical_in_mJy"],
                        data_selection["radio"],
                        s=120,
                        color=mark_color,
                        marker=marker,
                        label=i,
                        zorder=2
                    )
                    num -= 1

                elif i == "GRB140907A (Transient)":
                    ax1.scatter(
                        data_selection["optical_in_mJy"],
                        data_selection["radio"],
                        s=size_toplot,
                        color=mark_color,
                        marker=marker,
                        label=i,
                        zorder=num,
                        lw=2
                    )
                    ax1.arrow(
                        data_selection["optical_in_mJy"].iloc[0],
                        data_selection["radio"].iloc[0],
                        0, -0.05, fc=mark_color,
                        ec=mark_color,head_width=0.0045,
                        head_length=0.015, alpha=0.8
                    )
                else:
                    ax1.scatter(
                        data_selection["optical_in_mJy"],
                        data_selection["radio"],
                        s=size_toplot,
                        color=mark_color,
                        marker=marker,
                        label=i,
                        zorder=num,
                        lw=2
                    )
            num += 1
        # End of loop

        ax1.set_xlabel('Optical flux density (mJy)', size=label_size)
        ax1.set_ylabel('Radio flux density (mJy)', size=label_size)

        if not hide_arrow:
            extinction_arrow(
                ax1,
                OpticaltomJy(4., "R"),
                0.03,
                -(OpticaltomJy(4., "R") - OpticaltomJy(9., "R")),
                0,
                background,
                self.group_agn
            )

        ax1.tick_params(axis='both', which='major', labelsize=ticksize)
        ax6.tick_params(
            axis='both', which='major', labelsize=ticksize,
            zorder=20, direction='inout', length=6
        )

        # Hard code the limits
        ax1.set_xlim([0.00001,10000000])
        ax1.set_ylim([0.005,100000])

        zoomy2 = 2000   # used later for label placement

        if zoom_plot is not None:
            zoomx1, zoomy1, zoomx2, zoomy2 = zoom_plot.split(",")
            ax1.set_xlim([float(zoomx1), float(zoomx2)])
            ax6.set_xlim([float(zoomx1), float(zoomx2)])
            ax1.set_ylim([float(zoomy1), float(zoomy2)])

        # Draw the limit lines
        # TODO: Move these limits
        SDSSLimit = OpticaltomJy(22.2, "R")
        SDSSLimit_bright = OpticaltomJy(14.0, "R")
        Meerlicht_limit = OpticaltomJy(22.3, "R")
        MASTER_limit = OpticaltomJy(18.0, "R")
        LSST_limit = OpticaltomJy(24.0, "R")

        ax1.axvline(
            SDSSLimit, linestyle='--', color='k', linewidth=2.,
            zorder=1, alpha=0.6
        )
        ax1.axvline(
            SDSSLimit_bright, linestyle='--', color='k',
            linewidth=2., zorder=1, alpha=0.6
        )
        if not hide_line_labels:
            ax1.text(
                SDSSLimit_bright + (SDSSLimit_bright * 0.1),
                30000 * (float(zoomy2) / 400000),
                "SDSS Saturation Limit",
                rotation='vertical',
                weight='bold',
                size=16
            )
            ax1.text(
                SDSSLimit - (SDSSLimit * 0.35),
                40000 * (float(zoomy2) / 400000),
                "SDSS 95% Completeness",
                rotation='vertical',
                weight='bold',
                size=16
            )
            ax1.text(0.000012, 1-0.4, "FIRST Limit", weight='bold', size=16)
        if show_complete:
            first_red_line = Line2D(
                [SDSSLimit, 1e7],
                [1., 1.],
                color='red',
                linewidth=4,
                zorder=20
            )
            optical_line = Line2D(
                [SDSSLimit, SDSSLimit],
                [1., 5e5],
                color='red',
                linewidth=4,
                zorder=20
            )
            ax1.add_artist(first_red_line)
            ax1.add_artist(optical_line)

        ax1.axhline(
            1., linestyle='--', color='k', linewidth=2., zorder=1, alpha=0.6,
        )

        if meerkat:
            ax1.axvline(
                Meerlicht_limit,
                linestyle='--',
                color='#4daf4a',
                label="MeerLICHT Limit (5 min, 22.3 mag)",
                linewidth=4.,
                zorder=30
            )
            ax1.axhline(
                23.5e-3,
                linestyle='--',
                color='#1e90ff',
                label=r"5${\rm \sigma}$ MeerKAT Limit (1 h; 23.5 $\mu$Jy)",
                linewidth=4.,
                zorder=30
            )

        if ska_lsst:
            ax1.axvline(
                LSST_limit,
                linestyle='-',
                color='b',
                label="LSST Limit",
                linewidth=2.,
                zorder=1
            )
            ax1.axhline(
                2.1e-3,
                linestyle='-',
                color='k',
                label="SKA-1 Mid Limit",
                linewidth=2.,
                zorder=1
            )
            ax1.set_ylim([0.0001, 1600000])

        ax1.set_xscale('log')
        ax1.set_yscale('log')

        if push_agn:
            #Load in redshifts:
            redshifts = self._load_qso_redshifts()

            dmods = [np.sqrt(10.), 10.]

            labels = {
                0: "Quasars (Distance x 3.16)",
                1: "Quasars (Distance x 10)",
            }

            colors = {
                0: "orange",
                1: "lightblue",
            }

            data_selection = self.master_df[
                self.master_df['Type'] == 'Quasar'
            ].copy()

            data_selection = data_selection.loc[
                data_selection['Name'].isin(redshifts['name'])
            ]

            data_selection = data_selection.merge(
                redshifts[['name', 'z']], left_on='Name', right_on='name'
            )

            cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

            for i, d in enumerate(dmods):
                result = data_selection.apply(
                    kcorrect, axis=1, args=(d, i, cosmo)
                )

                ax1.scatter(
                    result[f"optical_dmod_{i}"],
                    result[f"radio_dmod_{i}"],
                    s=90,
                    color=colors[i],
                    marker=self.markers["Quasar"],
                    label=labels[i],
                    zorder=15 + i
                )

        if change_stellar_distanes:
            s_dist = self._load_stellar_distances()
            data_selection = self.master_df[
                (self.master_df['Type'].str.contains('Stellar')) &
                (self.master_df['Name'].isin(s_dist['object']))
            ]
            data_selection = data_selection.merge(
                s_dist, left_on='Name', right_on='object'
            )
            data_selection = data_selection[
                ['optical_in_mJy', 'radio', 'distance']
            ].apply(stellar_dist, axis=1, args=(push_stellar_dist,))

            ax1.scatter(
                data_selection['optical_pushed'],
                data_selection['radio_pushed'],
                s=50,
                color="#00BFFF",
                marker=self.markers["Stellar"],
                label=f"Stellar ({push_stellar_dist:.0f} pc)",
                zorder=16
            )

        if not schematic:
            handles, labels = ax1.get_legend_handles_labels()
            if (len(self.transients) != 0 or
                    push_agn or change_stellar_distanes):
                l2 = ax1.legend(
                    handles[len(self.basis):],
                    labels[len(self.basis):],
                    loc=2,
                    prop=self.prop,
                    scatterpoints=1,
                    markerscale=0.8
                )
            elif meerkat or ska_lsst:
                l2 = ax1.legend(
                    handles[:-len(self.basis):],
                    labels[:-len(self.basis)],
                    loc=2,
                    prop=self.prop,
                    scatterpoints=1,
                    markerscale=0.8
                )
            if qso_box:
                loca = 2
            else:
                loca = 1
            if not self.hide_legend:
                if meerkat or ska_lsst:
                    l1 = ax1.legend(
                        handles[-len(self.basis):],
                        labels[-len(self.basis):],
                        loc=loca,
                        prop=self.prop,
                        scatterpoints=1
                    ).set_zorder(30)
                else:
                    l1 = ax1.legend(
                        handles[:len(self.basis)],
                        labels[:len(self.basis)],
                        loc=loca,
                        prop=self.prop,
                        scatterpoints=1
                    ).set_zorder(30)
            if (len(self.transients)!=0 or push_agn or change_stellar_distanes
                or meerkat or ska_lsst):
                l2.set_zorder(30)
                ax1.add_artist(l2)

        self.current_fig = fig

        return fig

    def add_datapoint(
        self,
        name: str,
        o_mag: float,
        r_flux: float,
        o_band: str = 'R',
        ab: bool = True,
        marker='o',
        color="tab:blue",
        markersize: float = 100,
        **kwargs
    ) -> plt.figure:
        """Add a datapoint to the current figure.

        If not figure has been generated then a figure will be generated.
        Options for the plot can be declared using **kwargs.

        Args:
            name: The name of the object to add.
            o_mag: The optical magnitude.
            r_flux: The radio flux (mJy).
            o_band: The optical band of the magnitude.
            ab: Set to 'False' if the object magnitude is not in the AB system.
            marker (plt_markers): The matplotlib marker to use.
            color (plt_color): The matplotlib color to use.
            markersize: The size of the markers.
            **kwargs: Keyword arguments passed to 'generate_plot'.

        Returns:
            The current figure.
        """
        if self.current_fig is None:
            self.current_fig = self.generate_plot(**kwargs)

        ax1 = self.current_fig.get_axes()[0]

        if not ab:
            o_mag = ConvertToABMag(o_mag, o_band)

        o_flux = OpticaltomJy(o_mag, o_band)

        ax1.scatter(
            o_flux,
            r_flux,
            s=markersize,
            color=color,
            marker=marker,
            label=name,
            zorder=17
        )

        self.add_second_legend(ax1)

        return self.current_fig

    def add_datapoints_from_df(
        self,
        df: pd.DataFrame,
        ab_to_convert: List[str] = [],
        # Unsure about type hints for markers and colors?
        markers = [],
        colors = [],
        markersize: int = 100,
        group_by_type: bool = False,
        **kwargs
    ) -> plt.figure:
        """Add datapoints from a DataFrame to the current figure.

        Columns expected are:
            - 'Type'
            - 'Name'
            - 'radio'
            - 'optical_in_mJy'
            - 'R' or 'V'

        If not figure has been generated then a figure will be generated.
        Options for the plot can be declared using **kwargs.

        Args:
            df: The DataFrame containing the data points to add.
            ab_to_convert: The list of names that require converting to
                the AB system.
            markers: List of markers to use per type or per name. Make sure
                the number of markers match the number of unique names or
                types.
            colors: List of colors to use per type or per name. Make sure
                the number of colors match the number of unique names or
                types.
            markersize: The size to use for the markers.
            group_by_type: Group the datapoints by type rather than plotting
                with individual name labels.
            **kwargs: Keyword arguments passed to 'generate_plot'.

        Returns:
            The resulting current figure.
        """
        if self.current_fig is None:
            self.current_fig = self.generate_plot(**kwargs)

        for i in df['Name'].unique():
            if i not in ab_to_convert:
                self.ab_list.append(i)

        # easier for users to put in radio
        df = df.rename(columns={'radio': "F Radio / mJy"})

        df = self.master_table_analysis(df, transients=True)

        if group_by_type:
            unique = df['Type'].unique()
            unique_col = 'Type'
        else:
            unique = df['Name'].unique()
            unique_col = 'Name'

        if unique.shape[0] > len(markers):
            raise ValueError(
                'Number of markers is less than the number of unique entries.'
            )

        if unique.shape[0] > len(colors):
            raise ValueError(
                'Number of colors is less than the number of unique entries.'
            )

        ax1 = self.current_fig.get_axes()[0]

        for i, val in enumerate(unique):

            data_selection = df.loc[df[unique_col] == val]

            ax1.scatter(
                data_selection['optical_in_mJy'],
                data_selection['radio'],
                s=markersize,
                color=colors[i],
                marker=markers[i],
                label=val,
                zorder=17
            )

        self.add_second_legend(ax1)

        return self.current_fig

    def add_second_legend(self, ax1: plt.Axes) -> None:
        """
        Function to add the second legend to the plot when required.

        Args:
            ax1: The axis to add the legend to.

        Returns:
            None.
        """
        handles, labels = ax1.get_legend_handles_labels()
        l2 = ax1.legend(
            handles[len(self.basis):],
            labels[len(self.basis):],
            loc=2,
            prop=self.prop,
            scatterpoints=1,
            markerscale=0.8
        )

        if not self.hide_legend:
            l1 = ax1.legend(
                handles[:len(self.basis)],
                labels[:len(self.basis)],
                loc=1,
                prop=self.prop,
                scatterpoints=1
            ).set_zorder(30)


        l2.set_zorder(30)
        ax1.add_artist(l2)

    def add_text(
        self,
        text: str,
        x: float,
        y: float,
        **kwargs
    ) -> Tuple[plt.figure, plt.text]:
        """Adds text to the current figure.

        Args:
            text: The text string to add to the plot.
            x: The x coordinate of the text.
            y: The y coordinate of the text.
            **kwargs: Keyword arguments passed to 'plt.text'.
        """
        if self.current_fig is None:
            raise ValueError(
                'A plot must be generated before adding text!'
            )

        ax1 = self.current_fig.get_axes()[0]

        t = ax1.text(x, y, text, **kwargs)

        return self.current_fig, t

    def ratio_histogram(self) -> plt.figure:
        """Plot the ratio histogram.

        Returns:
            Histogram plot of the optical / radio ratios.
        """
        fig_hist = plt.figure(figsize=(6,15))
        subplots = {}
        basis_order = [
            "Stellar", "CV", "SN", "X-ray binary", "GRB",
            "Quasar (OPTICAL Sel.)", "Quasar (RADIO Sel.)", "Pulsar"
        ]
        legend_loc = {
            "Stellar":1, "CV":1, "SN":1, "X-ray binary":2, "GRB":2,
            "Quasar (OPTICAL Sel.)":2, "Quasar (RADIO Sel.)":2, "Pulsar":2
        }
        for i, val in enumerate(basis_order):
            label = "Radio Pulsar" if val == "Pulsar" else val
            subplots[val] = fig_hist.add_subplot(len(self.basis), 1, i+1)
            data_selection = self.base_data_df[
                self.base_data_df['Type'] == val
            ]
            subplots[val].hist(
                data_selection["ratio"],
                bins=np.logspace(-8, 6, 15),
                color=self.mark_colors[val],
                label=label,
                edgecolor="none"
            )

            subplots[val].set_xscale('log')
            subplots[val].legend(loc=legend_loc[val])
            subplots[val].grid(True)

            for label in subplots[val].xaxis.get_ticklabels()[::2]:
                label.set_visible(False)
            for label in subplots[val].yaxis.get_ticklabels()[::2]:
                label.set_visible(False)
            if val == "Pulsar":
                subplots[val].set_xlabel(
                    r"log $F_{\mathrm{r}}/F_{\mathrm{o}}$", size="x-large"
                )
            if val == "X-ray binary":
                subplots[val].set_ylabel("#", size="x-large")

        fig_hist.tight_layout()

        return fig_hist

    def frequency_histogram(self) -> plt.figure:
        """Plot the frequency histogram.

        Returns:
            Histogram plot of the radio frequencies used.
        """
        fig_freq = plt.figure(figsize=(9,6))
        ax = fig_freq.add_subplot(111)

        toplot=[
            "CV", "Pulsar", "X-ray binary", "SN", "GRB", "Stellar",
            "Quasar (RADIO Sel.)", "Quasar (OPTICAL Sel.)"
        ]
        toplot.reverse()

        labels=[
            "CV", "Radio Pulsar", "X-ray binary", "SN", "GRB",
            "Stellar", "Quasar (RADIO Sel.)", "Quasar (OPTICAL Sel.)"
        ]
        labels.reverse()

        histcolors=[self.mark_colors[i] for i in toplot]

        arraystoplot = []
        for i in toplot:
            filter_data = self.base_data_df[self.base_data_df["Type"] == i]
            arraystoplot.append(
                filter_data['F Radio Freq GHz'].to_numpy()
            )

        ax.hist(
            arraystoplot,
            bins=range(1,11,1),
            alpha=1.0,
            label=labels,
            color=histcolors,
            edgecolor="None"
        )

        ax.grid(True)
        ax.set_yscale('log')
        ax.set_ylim([0.1,1e5])

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize("large")

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize("large")

        ax.set_xlabel("Frequency (GHz)", size="x-large")
        ax.set_ylabel("#", size="x-large")
        ax.legend(prop={'size':12})

        return fig_freq

    def band_histogram(self) -> plt.figure:
        """Plot the optical band magnitudes.

        Returns:
            Histogram plot of the optical bands.
        """
        fig_band = plt.figure(figsize=(9,6))
        ax = fig_band.add_subplot(111)

        toplot=[
            "CV", "Pulsar", "X-ray binary", "SN", "GRB", "Stellar",
            "Quasar (RADIO Sel.)", "Quasar (OPTICAL Sel.)"
        ]
        toplot.reverse()

        labels=[
            "CV", "Radio Pulsar", "X-ray binary", "SN", "GRB",
            "Stellar", "Quasar (RADIO Sel.)", "Quasar (OPTICAL Sel.)"
        ]
        labels.reverse()

        histcolors = [self.mark_colors[i] for i in toplot]

        arraystoplot = []
        for i in toplot:
            filter_data = self.base_data_df[self.base_data_df["Type"] == i]
            arraystoplot.append(
                filter_data['optical_mag_used_band'].to_numpy()
            )

        ax.hist(
            arraystoplot,
            bins=[-0.5, 0.5, 1.5],
            histtype='bar',
            alpha=1.0,
            label=labels,
            color=histcolors,
            edgecolor="None"
        )

        ax.grid(True)
        ax.set_yscale('log')
        ax.set_xlabel("Optical Band", size="x-large")

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize("large")

        ax.set_ylabel("#", size="x-large")
        ax.set_ylim([0.1,1e5])
        ax.axvline(0.5, linestyle="--", color='k')
        ax.invert_xaxis()
        ax.legend(loc=2, prop={'size': 9})

        return fig_band

    def qso_z_histogram(self) -> plt.figure:
        """Plot the quasar redshift histogram.

        Returns:
            Histogram plot of the quasar redshift histogram.
        """
        redshifts = self._load_qso_redshifts()

        objectstoplot=["Quasar (OPTICAL Sel.)", "Quasar (RADIO Sel.)"]

        fig_qso_z = plt.figure(figsize=(9,6))
        ax = fig_qso_z.add_subplot(111)

        data_selection = self.base_data_df[
            self.base_data_df['Type'] == "Quasar (RADIO Sel.)"
        ].copy()
        data_selection['Name'] = data_selection['Name'].str.replace('PKS ', '')

        PKS_z = redshifts[
            redshifts['name'].isin(data_selection['Name'])
        ]['z'].to_numpy()

        data_selection = self.base_data_df.loc[
            self.base_data_df['Type'] == "Quasar (OPTICAL Sel.)"
        ]

        sdss_r9_mask = data_selection['Subtype'] == "SDSS_DR9"
        sdss_r9_data = data_selection[sdss_r9_mask]
        SDSS_r9_z = redshifts[
            redshifts['name'].isin(sdss_r9_data['Name'])
        ]['z'].to_numpy()

        sdss_r7_data = data_selection[~sdss_r9_mask]
        SDSS_r7_z = redshifts[
            redshifts['name'].isin(sdss_r7_data['Name'])
        ]['z'].to_numpy()

        histcolors=[
            self.mark_colors["Quasar (OPTICAL Sel.)"],
            self.mark_colors["CV"],
            self.mark_colors["Quasar (RADIO Sel.)"]
        ]

        toplot=[SDSS_r7_z, SDSS_r9_z, PKS_z]
        labels=[
            "Quasar (OPTICAL Sel.) SDSS DR7", "Quasar (OPTICAL Sel.) SDSS DR9",
            "Quasar (RADIO Sel.) PKS"
        ]

        ax.hist(
            toplot,
            bins=np.arange(0.0,7.5,0.5),
            histtype='bar',
            alpha=1.0,
            label=labels,
            edgecolor="None",
            color=histcolors,
            rwidth=0.95
        )

        ax.set_ylabel("#", size="x-large")
        ax.set_xlabel(r"$z$", size="x-large")
        ax.grid(True)
        ax.set_yscale('log')
        ax.legend(prop={'size':12})
        ax.set_ylim([0.06, 3000])

        return fig_qso_z

    def grb_z_histogram(self) -> plt.figure:
        """Plot the GRB redshift histogram.

        Returns:
            Histogram plot of the GRB redshifts.
        """
        grb_redshift_file = Path(pkg_resources.resource_filename(
            __name__, "./data/GRB_redshifts.txt"
        ))

        grb_redshifts = pd.read_csv(
            grb_redshift_file,
            names=['name', 'z'],
            dtype={'name': str, 'z': float}
        )

        data_selection = self.base_data_df.loc[
            self.base_data_df['Type'] == "GRB"
        ]

        usedgrbzeds = grb_redshifts[
            grb_redshifts['name'].isin(data_selection['Name'])
        ]['z'].to_numpy()

        fig_grb_z = plt.figure(figsize=(9,6))
        ax = fig_grb_z.add_subplot(111)
        ax.hist(
            usedgrbzeds,
            bins=np.arange(0.0,5.5,0.5),
            stacked=True,
            histtype='bar',
            alpha=1.0,
            label="GRBs",
            edgecolor="None",
            color=self.mark_colors["GRB"],
            rwidth=0.95
        )

        ax.set_ylabel("#", size="x-large")
        ax.set_xlabel(r"$z$", size="x-large")
        ax.grid(True)
        ax.legend(prop={'size':12})

        return fig_grb_z

    def stellar_distance_histogram(self) -> plt.figure:
        """Plot the stellar distance histogram.

        Returns:
            Histogram plot of the stellar distances.
        """
        if not self.group_stellar:
            raise ValueError(
                "RadioOpticalPlot must be initialised with"
                " 'group_stellar=True' to create the stellar distance"
                " histogram."
            )

        s_dist = self._load_stellar_distances()
        s_dist['distance_conv'] = s_dist['distance'] / 1.e3

        data_selection = self.base_data_df.loc[
            self.base_data_df['Type'] == "Stellar"
        ]

        s_distances = s_dist[
            s_dist['object'].isin(data_selection['Name'])
        ]['distance_conv'].to_numpy()

        fig_stellar_dist = plt.figure(figsize=(9,6))
        ax = fig_stellar_dist.add_subplot(111)
        ax.hist(
            s_distances,
            bins=np.arange(0.,17.,1.0),
            stacked=True,
            histtype='bar',
            alpha=1.0,
            label="Stellar sources",
            edgecolor="None",
            color=self.mark_colors["Stellar"],
            rwidth=0.95
        )

        ax.set_ylabel("#", size="x-large")
        ax.set_xlabel("Distance (kpc)", size="x-large")
        ax.grid(True)
        ax.set_yscale('log')
        ax.legend(prop={'size': 12})
        ax.set_ylim([0.8, 1000])

        return fig_stellar_dist

    def clear_current_fig(self) -> None:
        """Resets the current figure.

        Returns:
            None.
        """
        self.current_fig = None


@docstring_inherit(RadioOpticalPlot)
class RadioOpticalTrackPlot(RadioOpticalPlot):
    """The main plotting class for the Radio Optical plot with tracks.

    Attributes:
        packaged_tracks (Dict[str, str]): The names of objects with tracks
            available to plot.
        trackcolors (Dict[str, str]): What matplotlib colors to use for each
            class.
        trackmarkers (Dict[str, str]): What matplotlib markers to use for each
            class.
    """
    def __init__(
        self,
        base_data_file: Optional[str] = None,
        extra_exclude_stellar_types: Optional[List[str]] = None,
        extra_ab_list: Optional[List[str]] = None,
        group_agn: bool = False,
        group_stellar: bool = False,
        transients_file: Optional[str] = None
    ) -> None:
        """Init function.

        Args:
            base_data_file: The file containing the tab-separated master data.
                If 'None' is entered then the packaged latest master table
                will be used.
            extra_exclude_stellar_types: Extra stellar types to add to the
                stellar exclude list.
            extra_ab_list: Extra names to add to the AB list.
            group_agn: When 'True' the quasars are grouped together under
                the type 'Quasars'.
            group_stellar: When 'True' the stellar sources are all grouped
                under the type 'Stellar'.
            transients_file: Path to the transients file to load. Also accepts
                the names of the packaged transient files
                'transient_master_table_04072013.txt' and 'Stripe82_QSOs.txt'.

        Returns:
            None.
        """
        super().__init__(
            base_data_file, group_agn=group_agn, group_stellar=group_stellar,
            transients_file=transients_file
        )

        self.packaged_tracks = {
            "V404 Cyg": "XRB",
            "GX 339-4": "XRB",
            "GRB 970508": "GRB",
            "GRB 991208": "GRB",
            "GRB 000301C": "GRB",
            "GRB 030329": "GRB",
            "GRB 050820A": "GRB",
            "GRB 060218": "GRB",
            "GRB 070125": "GRB",
            "GRB 100418A": "GRB",
            "BL Lacertae": "Quasar",
            "BL Lacertae (3 month)": "Quasar",
            "3C 454.3": "Quasar",
            "SN 1990B": "SN",
            "SN 1993J": "SN",
            "SN 1994I": "SN",
            "SN 1998bw": "SN",
            "SN 2002ap": "SN",
            "SN 2004dj": "SN",
            "SN 2004dk": "SN",
            "SN 2004et": "SN",
            "SN 2004gq": "SN",
            "SN 2007bg": "SN",
            "SN 2007gr": "SN",
            "SN 2007uy": "SN",
            "SN 2008D": "SN",
            "SN 2008ax": "SN",
            "SN 2009bb": "SN",
            # "SN 2011dh": "GRB",
            "GRO J0422+32": "XRB",
            "GRO J1655-40": "XRB",
            "GS 1124-684": "XRB",
            # "GS 1354-64": "GRB",
            "XTE J1550-564": "XRB",
            "XTE J1859+226": "XRB",
            "XTE J0421+560": "XRB",
            "RS Oph": "CV",
            "SS Cyg": "CV",
            "T Pyx": "CV",
            "V1500 Cyg": "CV",
            "V1974 Cyg": "CV",
        }

        # Some attributes to manage the packaged tracks.
        self._cvs = [
            "SS Cyg", "RS Oph", "T Pyx", "V1500 Cyg", "V1974 Cyg"
        ]
        self._quasars = ["BL Lacertae", "BL Lacertae (3 month)", "3C 454.3"]
        # self._r_band_tracks = [
            # "GRB 030329", "GRB 100418A", "SN 1993J", "SN 1994I", "SN 1998bw"
        # ]
        self._track_ab_list = [
            "GRB 010921", "GRB 051221A", "GRB 080319B",
            "GRB 081203B", "SN 2008D"
        ]

    def list_packaged_tracks(self) -> None:
        """List the source tracks available to load.

        Returns:
            None.
        """
        pprint(self.packaged_tracks)

    def load_track_data(self, label: str) -> pd.DataFrame:
        """Load the track data from the packaged files.

        Args:
            label: The name of the source to load.

        Returns:
            The DataFrame containing the track data.
        """
        filenames = {
            "RS Oph": "CV_RSOph_6.0_V_45days.txt",
            "SS Cyg": "CV_SSCyg_8.6_V_15days.txt",
            "T Pyx": "CV_TPyx_5_V_445days.txt",
            "V1500 Cyg": "CV_V1500Cyg_8.1_V_373days.txt",
            "V1974 Cyg": "CV_V1974Cyg_5.0_V_862days.txt",
            "GRB 000301C": "GRB_000301C_8.46_R_43days.txt",
            "GRB 030329": "GRB_030329_8.64_R_65days.txt",
            "GRB 050820A": "GRB_050820A_8.46_R_25days.txt",
            "GRB 060218": "GRB_060218_8.46_R_23days.txt",
            "GRB 070125": "GRB_070125_8.46_R_18days.txt",
            "GRB 100418A": "GRB_100418A_8.46_R_31days.txt",
            "GRB 970508": "GRB_970508_8.46_R_82days.txt",
            "GRB 991208": "GRB_991208_8.46_R_37days.txt",
            "3C 454.3": "Quasar_3C454_8GHz.txt",
            "BL Lacertae (3 month)": "Quasar_BLLac_5GHz-3month.txt",
            "BL Lacertae": "Quasar_BLLac_5GHz.txt",
            "SN 1990B": "SN_1990B_5.0_V_105days.txt",
            "SN 1993J": "SN_1993J_8.3_R_305days.txt",
            "SN 1994I": "SN_1994I_8.3_R_127days.txt",
            "SN 1998bw": "SN_1998bw_8.64_R_77days.txt",
            "SN 2002ap": "SN_2002ap_1.43_R_17days.txt",
            "SN 2004dj": "SN_2004dj_5.0_V_110days.txt",
            "SN 2004dk": "SN_2004dk_8.5_R_41days.txt",
            "SN 2004et": "SN_2004et_5.0_R_57days.txt",
            "SN 2004gq": "SN_2004gq_8.5_R_35days.txt",
            "SN 2007bg": "SN_2007bg_8.46_R_68days.txt",
            "SN 2007gr": "SN_2007gr_4.9_R_92days.txt",
            "SN 2007uy": "SN_2007uy_8.4_R_56days.txt",
            "SN 2008D": "SN_2008D_4.8_R_115days.txt",
            "SN 2008ax": "SN_2008ax_8.46_V_46days.txt",
            "SN 2009bb": "SN_2009bb_8.46_V_44days.txt",
            "GRO J0422+32": "XRB_GROJ0422+32_5.0_V_155days.txt",
            "GS 1124-684": "XRB_GS1124-684_4.7_R_13days.txt",
            "GX 339-4": "XRB_GX339_4_9GHz_80days.txt",
            "GRO J1655-40": "XRB_J1655402005_4.86_V_45days.txt",
            "V404 Cyg": "XRB_V404_8_3GHz_35days.txt",
            "XTE J0421+560": "XRB_XTEJ0421+560_8.0_R_17days.txt",
            "XTE J1550-564": "XRB_XTEJ1550-564_4.8_V_9days.txt",
            "XTE J1859+226": "XRB_XTEJ1859+226_2.25_R_74days.txt",
        }

        try:
            file_name = filenames[label]
        except KeyError:
            raise KeyError(f'No packaged file found for object {label}!')

        track_file = Path(pkg_resources.resource_filename(
            __name__, f"./data/dynamic_tracks/{file_name}"
        ))

        track_df = pd.read_csv(track_file, comment="#", sep="\t")

        return track_df, track_file

    def _process_track_df(self, data_df: pd.DataFrame, t: str) -> pd.DataFrame:
        """Process the track data contained in the dataframe.

        Args:
            data_df: The DataFrame containing the track data.
            t: Source label (name).

        Returns:
            Processed track DataFrame.
        """
        if (np.any(data_df['R'].isna()) or np.any(data_df['R'] == 0.)):
            data_df['optical_mag_used_band'] = 'V'
            data_df['optical_mag_used_value'] = data_df['V']
        else:
            data_df['optical_mag_used_band'] = 'R'
            data_df['optical_mag_used_value'] = data_df['R']

        if t not in self._track_ab_list:
            data_df['optical_mag_used_value_processed'] = (
                data_df[[
                    'optical_mag_used_band',
                    'optical_mag_used_value'
                ]].apply(ConvertToABMag_pd, axis=1)
            )
        else:
            data_df['optical_mag_used_value_processed'] = (
                data_df['optical_mag_used_value']
            )

        data_df["optical_in_mJy"] = data_df[[
            "optical_mag_used_value_processed",
            "optical_mag_used_band"
        ]].apply(OpticaltomJy_pd, axis=1)

        return data_df

    def generate_track_plot(
        self,
        group_tracks: bool = False,
        only_tracks: Optional[List[str]] = None,
        exclude_tracks: Optional[List[str]] = None,
        only_types: Optional[List[str]] = None,
        summary_style: bool = False,
        start_end_only: bool = False,
        empty: bool = False,
        **kwargs
    ) -> plt.figure:
        """The main function to generate a Radio Optical plot with a track.

        Args:
            group_tracks: Set to 'True' to group the tracks into classes,
                instead of individual sources.
            only_tracks: List of sources to plot - no others will be plot.
            exclude_tracks: List of sources to exclude from plotting. Can not
                be used long with 'only_tracks'.
            only_types: List of types to plot - no sources from types outside
                of those listed will be plotted.
            summary_style: If 'True' the plot is returned in summary style
                where start and ends won't be circled.
            start_end_only: Only plot the start and end points of the tracks.
            empty: Generate an empty plot - useful to plot custom tracks.
            **kwargs: Keyword arguments passed to generate_plot.

        Returns:
            The resulting figure.
        """
        if only_tracks is not None and exclude_tracks is not None:
            warnings.warn(
                "Both only_tracks and exclude_tracks options have been"
                " used. Ignoring 'exclude_tracks'."
            )

            exclude_tracks = None

        if only_tracks is None and exclude_tracks is None:
            tracks_to_plot = self.packaged_tracks

        elif only_tracks is not None:
            tracks_to_plot = [
                i for i in self.packaged_tracks if i in only_tracks
            ]

        else:
            tracks_to_plot = [
                i for i in self.packaged_tracks if i not in exclude_tracks
            ]

        plot_all = summary_style

        if group_tracks and not summary_style:
            warnings.warn(
                'Summary style automatically selected when using group_tracks.'
            )
            plot_all = True

        if only_types is not None:
            for i in only_types:
                if i not in ['GRB', 'XRB', 'SN', 'CV', 'Quasar']:
                    raise ValueError(
                        f'{i} is not a valid class.'
                    )

            tracks_to_plot2 = []
            for i in tracks_to_plot:
                if self.packaged_tracks[i] in only_types:
                    tracks_to_plot2.append(i)

            tracks_to_plot = tracks_to_plot2

        if empty:
            tracks_to_plot = []


        if group_tracks:
            self.trackcolors = {
                "SN": self.mark_colors["SN"],
                "GRB": self.mark_colors["GRB"],
                "XRB": self.mark_colors["X-ray binary"],
                "CV": self.mark_colors["CV"],
                "BL Lacertae": "#87CEEB",
                "BL Lacertae (3 month)": "b",
                "3C 454.3": self.mark_colors["Quasar (RADIO Sel.)"],
                "Quasar": self.mark_colors["Quasar (RADIO Sel.)"],
            }

            self.trackmarkers = {
                "SN": self.markers["SN"],
                "GRB": self.markers["GRB"],
                "XRB": self.markers["X-ray binary"],
                "CV": self.markers["CV"],
                "BL Lacertae": "o",
                "BL Lacertae (3 month)": "o",
                "3C 454.3": "o",
                "Quasar": self.markers["Quasar (RADIO Sel.)"]
            }

        else:
            self.trackcolors = {
                "V404 Cyg": self.mark_colors["X-ray binary"],
                "GX 339-4": self.mark_colors["GRB"],
                "GRB 970508": "#6495ed",
                "GRB 991208": "#20b2aa",
                "GRB 000301C": "#dda0dd",
                "GRB 030329": self.mark_colors["GRB"],
                "GRB 050820A": "#3cb371",
                "GRB 060218": "#daa520",
                "GRB 070125": "#f08080",
                "GRB 100418A": self.mark_colors["Quasar (RADIO Sel.)"],
                "BL Lacertae": "#87CEEB",
                "BL Lacertae (3 month)": "b",
                "3C 454.3": self.mark_colors["Quasar (RADIO Sel.)"],
                "SN 1990B": "#6495ed",
                "SN 1993J": "#20b2aa",
                "SN 1994I": "#dda0dd",
                "SN 1998bw": self.mark_colors["GRB"],
                "SN 2002ap": "#3cb371",
                "SN 2004dj": "#daa520",
                "SN 2004dk": "#f08080",
                "SN 2004et": self.mark_colors["Quasar (RADIO Sel.)"],
                "SN 2004gq": "#ba55d3",
                "SN 2007bg": "#87CEEB",
                "SN 2007gr": '#ff8c00',
                "SN 2007uy": '#ee82ee',
                "SN 2008D": 'g',
                "SN 2008ax": '#4682b4',
                "SN 2009bb": '#9370d8',
                # "SN 2011dh": '#9acd32',
                "GRO J0422+32": "#6495ed",
                "GRO J1655-40": "#20b2aa",
                "GS 1124-684": "#dda0dd",
                "GS 1354-64": self.mark_colors["GRB"],
                "XTE J1550-564": "#3cb371",
                "XTE J1859+226": "#daa520",
                "XTE J0421+560": "#f08080",
                "RS Oph": "#6495ed",
                "SS Cyg": "#20b2aa",
                "T Pyx": "#dda0dd",
                "V1500 Cyg": self.mark_colors["GRB"],
                "V1974 Cyg": "#3cb371"
            }

            self.trackmarkers = {
                "V404 Cyg": 'd',
                "GX 339-4": 'p',
                "GRB 970508": "s",
                "GRB 991208": "o",
                "GRB 000301C": "v",
                "GRB 030329": 'D',
                "GRB 050820A": "^",
                "GRB 060218": "h",
                "GRB 070125": ">",
                "GRB 100418A": "H",
                "SN 1990B": "s",
                "SN 1993J": "o",
                "SN 1994I": 'v',
                "SN 1998bw": 'D',
                "SN 2002ap": "^",
                "SN 2004dj": "h",
                "SN 2004dk": ">",
                "SN 2004et": "H",
                "SN 2004gq": "<",
                "SN 2007bg": "d",
                "SN 2007gr": (4,1,0),
                "SN 2007uy": (4,1,45),
                "SN 2008D": (6,1,0),
                "SN 2008ax": (6,1,90),
                "SN 2009bb": (8,1,0),
                "SN 2011dh": (12,1,0),
                "BL Lacertae": "o",
                "BL Lacertae (3 month)": "o",
                "3C 454.3": "o",
                "GRO J0422+32": "s",
                "GRO J1655-40": "o",
                "GS 1124-684": "v",
                "GS 1354-64": "D",
                "XTE J1550-564": "^",
                "XTE J1859+226": "h",
                "XTE J0421+560": ">",
                "RS Oph": "s",
                "SS Cyg": "o",
                "T Pyx": "v",
                "V1500 Cyg": 'D',
                "V1974 Cyg": "^"
            }

        lightcurve_num = Counter()

        # allt_s = []
        fluxbins = {}
        lightcurves = {}
        alltracks_already_done = []

        # Generate background plot
        track_fig = self.generate_plot(**kwargs)

        ax1 = track_fig.get_axes()[0]


        for t in tracks_to_plot:
            data_df, file_name = self.load_track_data(t)

            # Skip 3 month if plotting all
            if plot_all and "(3 month)" in t:
                continue

            days = file_name.name.split("_")[-1].split(".")[0].split("days")[0]

            data_df["radio"] = data_df["RadioFlux"]

            if t == "GRB 050820A":
                data_df['radio'] = data_df['radio'] / 1.e3

            self._process_track_df(data_df, t)

            if "(3 month)" in t:
                zo = 16
            else:
                zo = 15

            if group_tracks:
                if "GRB" in t:
                    t = "GRB"
                elif "SN" in t:
                    t = "SN"
                elif t in self._cvs:
                    t = "CV"
                elif ("BL Lacertae" in t or t == "3C 454.3"):
                    t = "Quasar"
                else:
                    t = "XRB"

                lightcurve_num[t] += 1

                if start_end_only:
                    size = 600
                else:
                    size = 150
            else:
                size = 600

            if t in self.trackcolors:
                if not start_end_only:
                    ax1.plot(
                        data_df['optical_in_mJy'],
                        data_df['radio'],
                        color=self.trackcolors[t],
                        lw=3.0,
                        linestyle="--",
                        zorder=zo
                    )
                if not plot_all:
                    ax1.scatter(
                        data_df['optical_in_mJy'],
                        data_df['radio'],
                        marker=self.trackmarkers[t],
                        color=self.trackcolors[t],
                        s=size,
                        zorder=zo,
                        label=t+" ({0} days)".format(days)
                    )
                elif t not in alltracks_already_done:
                    if start_end_only:
                        ax1.scatter(
                            [
                                data_df['optical_in_mJy'].to_numpy()[0],
                                data_df['optical_in_mJy'].to_numpy()[-1]
                            ],
                            [
                                data_df['radio'].to_numpy()[0],
                                data_df['radio'].to_numpy()[-1]
                            ],
                            marker=self.trackmarkers[t],
                            color=self.trackcolors[t],
                            s=size,
                            zorder=zo,
                            label=t
                        )
                    else:
                        ax1.scatter(
                            data_df['optical_in_mJy'],
                            data_df['radio'],
                            marker=self.trackmarkers[t],
                            color=self.trackcolors[t],
                            s=size,
                            zorder=zo,
                            label=t
                        )
                    alltracks_already_done.append(t)
                else:
                    if start_end_only:
                        ax1.scatter(
                            [
                                data_df['optical_in_mJy'].to_numpy()[0],
                                data_df['optical_in_mJy'].to_numpy()[-1]
                            ],
                            [
                                data_df['radio'].to_numpy()[0],
                                data_df['radio'].to_numpy()[-1]
                            ],
                            marker=self.trackmarkers[t],
                            color=self.trackcolors[t],
                            s=size,
                            zorder=zo
                        )
                    else:
                        ax1.scatter(
                            data_df['optical_in_mJy'],
                            data_df['radio'],
                            marker=self.trackmarkers[t],
                            color=self.trackcolors[t],
                            s=size,
                            zorder=zo
                        )
            else:
                ax1.plot(
                    data_df['optical_in_mJy'],
                    data_df['radio'],
                    lw=1.5,
                    linestyle="--",
                    zorder=zo
                )
                ax1.scatter(
                    data_df['optical_in_mJy'],
                    data_df['radio'],
                    s=size,
                    zorder=zo,
                    label=t+" ({0} days)".format(days)
                )

            if t == "GRB 991208":
                data_df['Date'] = data_df['Date'] + 2

            if "BL Lacertae" not in t and "3C 454.3" not in t:
                addtracklabels(
                    data_df['optical_in_mJy'].to_numpy(),
                    data_df['radio'].to_numpy(),
                    ax1,
                    data_df['Date'].to_numpy(),
                    t.split(" ")[0],
                    plot_all,
                    t,
                    self._cvs,
                    start_end_only,
                    lightcurve_num[t],
                    t
                )

            lightcurves[t] = {
                "time":data_df['Name'],
                "radio":data_df['radio'],
                "optical":data_df['optical_in_mJy']
            }


        self.add_second_legend(ax1)

        self.current_fig = track_fig

        return self.current_fig


    def add_custom_track(
        self,
        df: pd.DataFrame,
        name: str,
        marker = "o",
        color = 'tab:blue',
        markersize: int = 150,
        ab: bool = True,
        **kwargs
    ) -> plt.figure:
        """Add a custom track from a DataFrame to the current figure.

        The required columns in the DataFrame are:
            - 'Name'
            - 'Date'
            - 'R' or 'V'
            - 'radio'

        If not figure has been generated then a figure will be generated.
        Options for the plot can be declared using **kwargs.

        Args:
            df: The DataFrame containing the custom track data.
            name: The name of the object. Will be used as the label.
            marker (maker): The marker to use for the track.
            color (color): The colour to use for the track.
            markersize: The size of the marker
            ab: Set to 'False' if the magnitude is not in the AB system.
            **kwargs: Keyword arguments passed to generate plot there is no
                current plot.

        Returns:
            The resulting custom track figure.
        """

        if self.current_fig is None:
            self.current_fig = self.generate_plot(**kwargs)

        if ab:
            self._track_ab_list.append(name)

        data_df = self._process_track_df(df, name)

        days = data_df['Date'].iloc[-1] - data_df['Date'].iloc[0]

        ax1 = self.current_fig.get_axes()[0]

        ax1.plot(
            data_df['optical_in_mJy'],
            data_df['radio'],
            color=color,
            lw=3.0,
            linestyle="--",
            zorder=17
        )

        ax1.scatter(
            data_df['optical_in_mJy'],
            data_df['radio'],
            marker=marker,
            color=color,
            s=markersize,
            zorder=17,
            label=name+" ({0} days)".format(days)
        )

        addtracklabels(
            data_df['optical_in_mJy'].to_numpy(),
            data_df['radio'].to_numpy(),
            ax1,
            data_df['Date'].to_numpy(),
            name.split(" ")[0],
            False,
            name,
            self._cvs,
            False,
            1,
            name,
            zorder=18
        )

        self.add_second_legend(ax1)

        return self.current_fig
