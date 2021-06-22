# Radio Optical Transients Plotting Package

This package contains all the code required to reproduce, and interact with, the plots found in the [Stewart et al. 2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.2481S/abstract) publication *On the optical counterparts of radio transients and variables*.

![Figure 1](https://github.com/4pisky/radio-optical-transients-plot/blob/main/examples/ro_figure_1.png)

## Notes

A few notes on the package:

* The original code to produce the plots began life 10 years ago. 
    This packaged version is a limited time attempt to update the code to a somewhat modern state and to make it more accessible.
* Some of the code may still appear to be outdated with unnecessary loops or overlong plotting statements. 
    The package was created by butchering the original, single script, code so some old code still remains.
* There are quite a few 'hard codes' in the code, such as markers and colors for different object types, and flags to avoid data issues.
* The formats of entry files is somewhat a pain. They are tab-separated with a header of:
    ```Name	Type	Subtype	F X-Ray	F X-Ray Error	B	B Error	V	V Error	R	R Error	I	I Error	F IR	F IR Error	F Radio / mJy	F Radio Error	F Radio Freq GHz	T|R-O| (UT)```
    This is a leftover from our original ambition to collect X-ray, IR and all optical fluxes.
    Pandas is used in this updated version so it's possible to only included required columns, however new functions have been written to allow the addition of data via DataFrames (see the notebook example).
    Track inputs are slightly different, see the section below.
* It is designed to also be notebook friendly.
* Please refer to the paper for details on the data itself.

## Installation

It is recommended to create a virtual environment using your chosen virtual environment manager.
Clone the repoistory from GitHub and install using `pip`:

```
git clone https://github.com/4pisky/radio-optical-transients-plot.git
cd radio-optical-transients-plot
pip install .
```

The package was developed on and tested with Python 3.9.5.

### Dev Installation

This package uses [poetry](https://python-poetry.org) as the package manager.
For development work install the package using poetry:

```
cd radio-optical-transients-plot
poetry install
```

By default this will install all the dependancies including the dev-only deps.

## Usage

There are three main classes included in `radio_optical_transients_plot.main`:

* `RadioOpticalData()` manages the data reading and processing/sorting.
* `RadioOpticalPlot()` contains the main functions to produce the standard plots. Inherits from `RadioOpticalData()`.
* `RadioOpticalTrackPlot()` builds upon `RadioOpticalPlot` to allow track plotting as seen in the paper. Inherits from `RadioOpticalPlot()`.

Unfortunately there is not full documentation but please refer to the docstrings and also two files that show example usage:

### ro\_notebook\_example.ipynb

This file is found in `examples`. The notebook contains extensive examples of how to use the package, including how to add custom datapoints or tracks to the plot.

### radio\_optical\_figures.py

This script, installed with the package and accessible from the command line, contains the code usage required to reproduce every plot found in the original paper.

```
radio_optical_figures -h
usage: radio_optical_figures [-h] --figure {1,2,3,4,5a,5b,6a,6b,6c,6d,7,8a,8b,9,a1a,a1b,a2,a3,a4} [-s]
                             [-f {png,pdf}]

optional arguments:
  -h, --help            show this help message and exit
  --figure {1,2,3,4,5a,5b,6a,6b,6c,6d,7,8a,8b,9,a1a,a1b,a2,a3,a4}
                        Specify what figure to produce. (default: None)
  -s, --save            Use this option to automatically save the figures instead of displaying. (default: False)
  -f {png,pdf}, --save-format {png,pdf}
                        Select the format the figure will be saved as. (default: png)
```

Example usage:

```
radio_optical_figures --figure 1 -s
```

## Tracks

Plotting tracks is relatively straight forward, the data input format is tab-separated with the following header:

```
Name	Date	B	B Error	V	V Error	R	R Error	I	I Error	RadioFlux	RadioFlux Error	RadioFreq
```

Note here that while `Date` was originally intended to hold the actual date, it changed to meaning the **number of days**. Hence why some entries appear as so:

```
Name	Date	B	B Error	V	V Error	R	R Error	I	I Error	RadioFlux	RadioFlux Error	RadioFreq
GRB 000301C	4.26	0	0	0	0	20.86	0.04	0	0	0.316	0.041	8.46
GRB 000301C	4.98	0	0	0	0	21.43	0.26	0	0	0.289	0.034	8.46
GRB 000301C	12.17	0	0	0	0	23.1	0.22	0	0	0.483	0.026	8.46
GRB 000301C	14.17	0	0	0	0	23.82	0.1	0	0	0.312	0.062	8.46
GRB 000301C	34.18	0	0	0	0	26.5	0.15	0	0	0.325	0.027	8.46
GRB 000301C	48.06	0	0	0	0	27.9	0.15	0	0	0.145	0.036	8.46
```

Please refer to the notebook example on how to add tracks by just using a DataFrame and bypassing the formatting requirements.

*Note: Come to think of it I don't think there is a function to upload a new track direct from the file like this. Use the pandas method!*

## Packaged Data

Some data files are packaged with the code (found in `radio_optical_transients_plot/data`):

* `dynamic_tracks`: contains all the data files for the included tracks in the paper.
* `all_qso_redshifts`: contains the redshift information for the quasars.
* `GRB_redshifts.txt`: contains the redshift information for the GRBs.
* `Master_Table_27042018.tsv`: the main basis data file.
* `stellar_distances.csv`: contains the distance information for the stellar sources.
* `Stripe82_QSOs.txt`: contains the Stripe 82 quasars to overplot.
* `transient_master_table_04072013.txt`: contains the FIRST transients to overplot.

## Tests

A few tests have been written to test the conversion functions, however substantial testing is not present.