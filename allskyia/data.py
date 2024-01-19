import os

import cv2 as cv
import numpy as np
from astropy.io import fits
from astropy.time import Time
import astropy.units as u


class AllskyData:
    """
    Allsky Data
    """

    def __init__(self, filename: str):
        self.data = None
        self.bgr_data = None
        self.header = None
        self.obstime = None
        self.obstime_utc = None
        self.load_data(filename)

    def load_data(self, filename: str) -> bool:
        """
        Load data from a file.
        Args:
            filename ():

        Returns:

        """
        if filename.endswith('.fits') or filename.endswith('.fits.bz2'):
            return self.load_fits(filename)
        elif filename.endswith('.jpg'):
            return self.load_jpeg(filename)
        else:
            return False

    def load_fits(self, filename: str) -> bool:
        """
        Load a FITS file.
        Parameters
        ----------
        filename
            The file name of a FITS file.
            *.fits or *.fits.bz2

        Returns
        -------
        bool
            True if the file is loaded successfully.

        """

        if not os.path.isfile(filename):
            return False

        # Try to load the file for 100 times.
        for i in range(100):
            try:
                data, self.header = fits.getdata(filename, header=True)
                # FITS IS BIG ENDIAN

                self.obstime = Time(self.header['DATE-OBS'])
                self.obstime_utc = self.obstime - 8 * u.hour  # convert to UTC+0
                break
            except:
                pass

        # transpose data from (color, y, x) to (y, x, color)
        self.bgr_data = data.copy().astype(float)
        # color convert from rgb to bgr
        self.bgr_data = self.bgr_data[[2, 1, 0]]
        self.bgr_data = np.swapaxes(self.bgr_data, 0, 1)
        self.bgr_data = np.swapaxes(self.bgr_data, 1, 2)
        self.bgr_data -= self.bgr_data.min()
        self.bgr_data /= self.bgr_data.max()

        # move origin from top left to bottom left
        self.bgr_data = np.flip(self.bgr_data, axis=0)
        self.bgr_data = np.ascontiguousarray(self.bgr_data)

        # Load RGB three colors and convert to brightness
        data = data.astype(float)
        data = data - data.min()
        data = data[0] + data[1] + data[2]
        data /= 3

        # flip
        data = np.flip(data, axis=0)
        data = np.ascontiguousarray(data)
        self.data = data

        return True

    def load_jpeg(self, filename: str) -> bool:
        """
        Load a JPEG file.
        The observation time (obstime) uses the file modified time.
        Args:
            filename ():

        Returns:

        """
        data = cv.imread(filename)
        data = np.flip(data, axis=0)
        data = np.ascontiguousarray(data)

        # read create time of file
        self.obstime_utc = Time(os.path.getmtime(filename), format='unix')
        self.obstime = self.obstime_utc + 8 * u.hour  # convert to UTC+8

        self.data = data.astype(float)
        self.data = self.data.mean(axis=-1)
        self.bgr_data = data.copy().astype(float) / 255.

        return True
