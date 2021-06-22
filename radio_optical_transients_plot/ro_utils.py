"""Radio Optical utility functions.

This script contains utility functions used by the RadioOptical plotting
package.

This file can be imported as a module and contains the following
functions:

    * get_zeropoint - get the zeropoint for an optical band.
    * OpticaltomJy - Converts optical magnitude to mJy.
    * OpticaltomJy_pd - Pandas .apply wrapper function for OpticaltomJy.
    * mJytoOptical - Converts radio flux to optical magnitude.
    * ConvertToABMag - Converts a magnitude to the AB system.
    * ConvertToABMag_pd - Pandas .apply wrapper function for ConvertToABMag.
    * stellar_dist - Calculates new fluxes when the stellar sources are pushed
        back to a further distance.
    * kcorrect_calc - Calculation of the k-correction.
    * kcorrect - Calculates quasars new fluxes when pushed back taking into
        account the k-correction.
"""

import numpy as np
import pandas as pd

from astropy.coordinates import Distance
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy import units as u


def get_zeropoint(band: str) -> float:
    """
    Defines the zeropoints for the optical bands used for conversions.

    Args:
        band: The band for which to obtain the zero point.

    Returns:
        The zero point value.

    Raises:
        ValueError: If the band is not one of 'R', 'V', 'B' or 'I'.
    """

    # If the below values are changed, the tests must be updated!
    zeropoints = {
        "R": 3631.,
        "V": 3631.,
        "B": 4130.,
        "I": 2635.
    }

    if band not in zeropoints:
        raise ValueError(f"Zero point not available for band {band}.")

    return zeropoints[band]


def OpticaltomJy(mag: float, band: str) -> float:
    """Converts an optical magnitude of a given band to mJy.

    Args:
        mag: The optical magnitude.
        band: The band of the optical magnitude provided.

    Returns:
        The optical flux converted to mJy.
    """
    zeropoint = get_zeropoint(band)
    diminish = 10.**(-0.4 * mag)
    flux = zeropoint * diminish
    mflux = flux * 1000  # Convert to mJy

    return mflux


def OpticaltomJy_pd(row: pd.Series) -> float:
    """Wrapper function to use with pd.DataFrame.apply to convert the
    optical magnitude to mJy.

    Requires the columns 'optical_mag_used_value_processed' and
    'optical_mag_used_band'.

    Args:
        row: The pandas series containing the required columns.

    Returns:
        The converted flux.
    """
    mag = row["optical_mag_used_value_processed"]
    band = row["optical_mag_used_band"]

    mflux = OpticaltomJy(mag, band)

    return mflux


def mJytoOptical(mflux: float, band: str):
    """Converts a radio flux in mJy to an optical band.

    Args:
        mflux: The radio flux in mJy.
        band: The optical band to use to convert to magnitudes.

    Returns:
        The radio flux in optical magnitudes.
    """
    flux = mflux / 1000.

    zeropoint = get_zeropoint(band)

    mag = (np.log10((flux / zeropoint))) / -0.4

    return mag


def ConvertToABMag(mag: float, band: str) -> float:
    """Converts a magnitude to the AB system.

    Args:
        mag: The magnitude to convert.
        band: The band of the magnitude.

    Returns:
        The magnitude converted to the AB system.

    Raises:
        ValueError: If the band is not 'R' or 'V'.
    """
    if band == "V":
        return mag - 0.044
    elif band == "R":
        return mag + 0.055
    else:
        raise ValueError(f"Band {band} cannot be converted to AB.")


def ConvertToABMag_pd(
    row: pd.Series,
    band_row: str = "optical_mag_used_band",
    mag_row: str = "optical_mag_used_value"
) -> float:
    """Wrapper function to use with pd.DataFrame.apply to convert the
    optical magnitude to the AB system.

    Requires the columns 'optical_mag_used_value' and
    'optical_mag_used_band'.

    Args:
        row: The pd.Series object containing the required columns.
        band_row: The name of the column that contains the band.
        mag_row: The name of the column that contains the magnitude values.

    Returns:
        The converted magnitude.
    """
    band = row[band_row]
    mag = row[mag_row]

    mag = ConvertToABMag(mag, band)

    return mag


def stellar_dist(row: pd.Series, push: float) -> pd.Series:
    """Calculate the fluxes for the stellar objects at a given distance.

    For use with pandas.DataFrame.apply(). Requires the columns:
        - 'distance'
        - 'radio'
        - 'optical_in_mJy'

    Args:
        row: The pd.Series object containing the required rows.
        push: The distance in pc to push the stellar objects.

    Returns:
        The pandas series with the new calculated fluxes in the columns
        'radio_pushed' and 'optical_pushed'.
    """
    current_dist = row['distance']
    r_flux = row['radio']
    o_flux = row['optical_in_mJy']
    newdist = current_dist + push

    row['radio_pushed'] = (
        r_flux * (current_dist * current_dist)) / (newdist * newdist)
    row['optical_pushed'] = (
        o_flux * (current_dist * current_dist)) / (newdist * newdist)

    return row


def kcorrect_calc(
    d: float, oldflux: float, z1: float, z2: float, a: float
) -> float:
    """Performs the k-correction calculation given the required arguments.

    Args:
        d: The distance modulation, in terms of a factor. E.g. a value of 10
            would indicate a distance change of 10x.
        oldflux: The current flux of the source (radio or optical).
        z1: The current redshift.
        z2: The new redshift.
        a: The alpha value to use in the calculation.

    Returns:
        The k-corrected new flux.
    """
    return oldflux * ((1. / d) * (1. / d)) * ((1 + z1) / (1 + z2))**(-(a) - 1)


def kcorrect(
    row: pd.Series, dmod: float, n: int, cosmo: FlatLambdaCDM
) -> pd.Series:
    """Perform a k-correction on the Quasar dataframe with a given distance
    modulation.

    For use with pd.DataFrame.apply() and requires the columns:
        - 'radio'
        - 'optical_to_mJy'
        - 'z'

    Args:
        row: The pd.Series with the required columns.
        dmod: The distance modulation, in terms of a factor. E.g. a value of 10
            would indicate a distance change of 10x.
        n: Column label counter.
        cosmo: The astropy.cosmology cosmology object.

    Returns:
        The pandas series with the new calculated fluxes in the columns
        'radio_dmod_\{n\}' and 'optical_dmod_\{n\}'.
    """
    ralpha = -0.7
    oalpha = -0.44

    z1 = row['z']
    d1 = cosmo.luminosity_distance(z1)
    rflux = row['radio']
    oflux = row['optical_in_mJy']

    d2 = d1.value * dmod
    d2 = Distance(d2, u.Mpc)
    z2 = z_at_value(cosmo.luminosity_distance, d2)
    newradio = kcorrect_calc(dmod, rflux, z1, z2, ralpha)
    newoptical = kcorrect_calc(dmod, oflux, z1, z2, oalpha)

    row[f'radio_dmod_{n}'] = newradio
    row[f'optical_dmod_{n}'] = newoptical

    return row
