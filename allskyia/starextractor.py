import numpy as np
import pandas as pd
import sep


def extract_stars_sep(data, mask, extract_treshold=3, bw=16, bh=16):
    bkg = sep.Background(data, mask, bw=bw, bh=bh)
    data_sub = data - bkg
    objects = sep.extract(data_sub, extract_treshold, mask=mask, err=bkg.rms())
    extracted_stars = pd.DataFrame(objects)
    print(f'{len(extracted_stars)} stars are extracted')

    # Alert! Not using aperture photometry. Use what extract() returns. `cflux` is what I need.
    # flux, fluxerr, flag = sep.sum_circle(data_sub, objects['x'], objects['y'],
    #                                      aperture_radius, err=bkg.globalrms, gain=1.0)

    extracted_stars.rename(columns={'x': 'extracted_x', 'y': 'extracted_y'}, inplace=True)

    # calculate signal-to-noise ratio
    extracted_stars['snr'] = extracted_stars['flux'] / np.sqrt(
        extracted_stars['flux'] + bkg.globalback * extracted_stars['npix'])

    return extracted_stars
