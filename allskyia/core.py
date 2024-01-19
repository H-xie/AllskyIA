import astropy.units as u
import cv2 as cv
import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, SkyCoord
from astropy_healpix.healpy import ang2pix

from allskyia.camera import Calibrate
from allskyia.data import AllskyData
from allskyia.photometry import Photometry
from allskyia.starextractor import extract_stars_sep
from allskyia.starmatch import StarMatcher
from allskyia.utils import alt_to_airmass


def create_mask(shape, center, radius):
    assert len(center) == len(shape) == 2

    for iShape in range(len(shape)):
        assert 0 <= center[iShape] < shape[iShape]
        assert 0 <= radius < shape[iShape] / 2

    center = tuple(map(int, center))

    # 创建一个空白的 mask
    mask = np.zeros(shape, dtype=np.uint8)
    # 画一个圆
    cv.circle(mask, center, int(radius), 255, -1)

    mask = mask.astype(bool)
    mask = np.logical_not(mask)
    return mask


class Allsky:
    BACKGROUND_BOX = 16
    BACKGROUND_FILTER = 3

    def __init__(self, location, catalog: pd.DataFrame, image_shape, debug_mode: bool = False):
        self.pair = None
        self.location = location
        self.catalog = None
        self.photometric_model: Photometry = None
        self.cali_model = None
        self.obstime_utc = None
        self.obstime = None
        self.data_without_background = None
        self.sources = None
        self.mask = None
        self.header = None
        self.data = None
        self.bgr_data = None
        self.tycho_connection = None
        self.debug = debug_mode
        self.image_shape = image_shape
        self.original_catalog = catalog.copy()

    def load_data(self, filename: str) -> bool:

        asd = AllskyData(filename)
        self.data = asd.data
        self.bgr_data = asd.bgr_data
        self.header = asd.header
        self.obstime = asd.obstime
        self.obstime_utc = asd.obstime_utc

    # create mask by centre and radius
    def create_mask(self, centre: tuple, radius: int | float, shape: tuple | None = None) -> np.ndarray:
        """
        Create a mask by centre and radius.
        Parameters
        ----------
        centre
            The centre of the mask.
        radius
            The radius of the mask.
        shape
            The shape of the mask. If None, use the shape of the FITS file.
        Returns
        -------
        np.ndarray
            The mask.

        """
        if shape is None:
            if self.data is None:
                raise "No FITS file loaded."
            else:
                shape = self.data.shape

        self.mask = create_mask(shape, centre, radius)

        return self.mask

    def extract_stars_sep(self, extract_treshold=3):
        extracted_stars = extract_stars_sep(self.data, self.mask,
                                            extract_treshold,
                                            bw=self.BACKGROUND_BOX,
                                            bh=self.BACKGROUND_BOX)
        self.sources = extracted_stars
        return extracted_stars

    def star_match(self, extracted_stars: pd.DataFrame = None, *args, **kwargs) -> pd.DataFrame:
        if extracted_stars is None:
            extracted_stars = self.sources
        matcher = StarMatcher(*args, **kwargs)
        self.pair: pd.DataFrame = matcher.match(self.catalog, extracted_stars)

        return self.pair

    def set_calibrate_model(self, model: Calibrate):
        assert model.is_loaded is True
        self.cali_model = model

    def catalog_to_photo(self, cali_model=None, horizon=0 * u.deg):

        if cali_model is None:
            cali_model = self.cali_model

        assert cali_model is not None

        self.catalog = self.original_catalog.copy()

        stars = SkyCoord(ra=self.catalog.RA_ICRS_, dec=self.catalog.DE_ICRS_, unit="deg")
        altaz = stars.transform_to(AltAz(obstime=self.obstime_utc, location=self.location))
        self.catalog['alt'] = altaz.alt.value
        self.catalog['az'] = altaz.az.value
        self.catalog = self.catalog[self.catalog['alt'] >= horizon.value].reset_index(drop=True)
        print(f'the number of rise stars: {len(self.catalog)}')

        alt = self.catalog['alt'].values
        az = self.catalog['az'].values
        result = cali_model.forward(alt, az)
        self.catalog['calculated_x'] = result[:, 0]
        self.catalog['calculated_y'] = result[:, 1]

        return self.catalog

    def set_photometric_model(self, model: Photometry):
        self.photometric_model = model

    def extinction(self, pair: pd.DataFrame, threshold, hp, photometric_model=None):
        assert 'alt' in pair.columns
        assert 'VTmag' in pair.columns
        assert 'flux' in pair.columns

        if photometric_model is None:
            photometric_model = self.photometric_model

        assert photometric_model is not None

        # using nan_flux to fill nan value in flux
        # This indicates that the stars are invisible in the image
        # pair['flux'] = pair['flux'].fillna(nan_flux)

        airmass = alt_to_airmass(pair['alt'].values)

        flux_standard = photometric_model.flux(pair['VTmag'].values, airmass)  # 标准星光通量
        extinction = (flux_standard - pair['flux'])  # 消光程度

        # pair['extinction'] = extinction
        pair['flux_standard'] = flux_standard

        result = pair[['alt', 'az', 'flux', 'flux_standard']].copy()
        result.loc[extinction < threshold, 'flux'] = flux_standard[
            extinction < threshold]  # 消光过大的，认为是测量误差，直接用标准星光通量代替

        result = self.healpix_extinction(result, hp, threshold)
        return result

    def healpix_extinction(self, data: pd.DataFrame, hp, horizon=20 * u.deg):

        data['hp_index'] = hp.lonlat_to_healpix(data['az'].values * u.deg,
                                                data['alt'].values * u.deg)

        flux = data[['hp_index', 'flux']]
        flux = flux.groupby('hp_index').sum()  # NAN is treated as 0.
        flux.sort_index(inplace=True)

        flux_standard = data[['hp_index', 'flux_standard']].groupby('hp_index').sum()
        flux_standard.sort_index(inplace=True)

        extinction = 1 - (flux['flux'] / flux_standard['flux_standard'])

        hp_data = pd.DataFrame(index=range(hp.npix), columns=['extinction'], dtype=np.float64)
        hp_data['extinction'] = np.nan

        max_hpindex = ang2pix(hp.nside, np.pi / 2 - np.deg2rad(horizon.value),
                              np.pi * 2 - (np.pi / 4 / hp.nside))  # 最大的 HEALPix 索引
        extinction = extinction[extinction.index <= max_hpindex]
        hp_data.loc[extinction.index, 'extinction'] = extinction.loc[extinction.index]
        print(np.isnan(extinction).sum(), ' HEALPix zone is nan')

        return hp_data

    def visibility(self, hp, horizon=0 * u.deg):

        if self.pair is None:
            # create mask
            self.create_mask(self.center, self.radius(horizon.value))
            # extract stars and match
            self.extract_stars_sep()
            self.catalog_to_photo(horizon=horizon)
            self.star_match(distance_upper_bound=2)

        pair = self.pair

        assert 'VTmag' in pair.columns

        pair['hp_index'] = hp.lonlat_to_healpix(pair['az'].values * u.deg,
                                                pair['alt'].values * u.deg)

        # get the faintest magnitude of each HEALPix zone
        standard_magnitude = pair[['hp_index', 'VTmag']]
        standard_magnitude = standard_magnitude.groupby('hp_index').max()
        standard_magnitude.sort_index(inplace=True)

        # select extracted stars
        star_index = np.logical_not(np.isnan(pair['flux']))
        magnitude = pair.loc[star_index, ['hp_index', 'VTmag']]
        magnitude = magnitude.groupby('hp_index').max()
        magnitude.sort_index(inplace=True)

        # scale magnitude to score, ranging from 0 to 1
        magnitude['visibility'] = (2 + magnitude['VTmag']) / (2 + standard_magnitude['VTmag'])

        max_hpindex = ang2pix(hp.nside, np.pi / 2 - np.deg2rad(horizon.value),
                              np.pi * 2 - (np.pi / 4 / hp.nside))  # 最大的 HEALPix 索引
        hp_data = pd.DataFrame(index=range(hp.npix), columns=['visibility'], dtype=np.float64)
        hp_data['visibility'] = np.nan
        hp_data[(hp_data.index <= max_hpindex)] = 0
        hp_data.loc[magnitude.index, 'visibility'] = magnitude.loc[magnitude.index]

        return hp_data

    @property
    def north_theta_rad(self):
        return np.deg2rad(-self.cali_model.directions()['north_theta'])

    @property
    def center(self):
        return self.cali_model.center

    def radius(self, alt):
        return self.cali_model.radius(alt)
