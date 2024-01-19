# Allsky Image Analysis (allskyia)

This is a python package for allsky image analysis. It can be used to calibrate the allsky camera and calculate the visibility of the stars. The project is developed as a Python package. 


## Sample codes
### Camera calibration

Befor calculate visibility, you need to get camera model first to transform AltAz coordinates to its photo coordinates. The camera model can be trained by the following codes.

Note: codes is incomplete. You need to fill in the location of the allsky camera, the image shape, the center point and radius of the image, the north direction in the image, and the path of the image you want to use for calibration. 

```python
# This file is used to calibrate the allsky camera.
# The photo coordinates of the stars in the catalog are calculated by the rough coefficients.
# Loading catalog. Here we use Tycho-2 catalog.
import astropy.units as u
import pandas as pd
import sqlalchemy
import torch
from astropy.coordinates import EarthLocation, AltAz, SkyCoord

from allskyia import AllskyData, StarMatcher, Calibrate, altaz_to_photo, draw_stars, draw_catalog, create_mask, extract_stars_sep

# Loading catalog. Here we use Tycho-2 catalog.
# `catalog_full` is a pandas DataFrame object. The column names keep the same as the original catalog.
tycho_engin = sqlalchemy.create_engine('DATABASE URL')
tycho_engin.connect()
catalog = pd.read_sql_query(f"SELECT * FROM TableName WHERE VTmag <= 4 ORDER BY VTmag", tycho_engin)

# Fill in the location of the allsky camera. 
# You can use EarthLocation.from_site('SITE NAME') as well
location = EarthLocation.from_geodetic(lon=, lat=, height=)

# 
image_shape = (rows, cols)  # Replace with the actual image shape
center = (U, V)             # Replace with the zenith position 
radius = R                  # Replace with the distance from the zenith to horizon in unit of pixel
north_theta = NORTH * u.deg  # Replace with the north direction in the image. 

asd = AllskyData('IMAGE PATH')  # Replace IMAGE PATH with the path of the image you want to use for calibration.
mask = create_mask(image_shape, center, radius / 90 * 70)  # keep altitude > 20
extracted_stars = extract_stars_sep(asd.data, mask)
# get most bright 200 stars
extracted_stars = extracted_stars.sort_values(by='flux', ascending=False).iloc[:200]
draw_stars(asd.bgr_data * 255, extracted_stars, 'stars.png')    # save image to stars.png

# get AltAz
frame = AltAz(location=location, obstime=asd.obstime_utc)
altaz = SkyCoord(ra=catalog['RA_ICRS_'], dec=catalog['DE_ICRS_'], unit=u.deg).transform_to(frame)
catalog['alt'] = altaz.alt.value
catalog['az'] = altaz.az.value

# Define a tensor for transformation
_A = torch.tensor([[-1 / 90, 0, 1],
                   [0, 1, north_theta.value],
                   [0, 0, 1]],
                  dtype=torch.float)

# Define the radius and center point in pixels
s = radius  # radius (unit: pixels)
x0, y0 = center  # center point (unit: pixels)
_B = torch.tensor([[s, 0, x0],
                   [0, s, y0]],
                  dtype=torch.float)
xy = altaz_to_photo(alt=altaz.alt.value, az=altaz.az.value, A=_A, B=_B)
xy = xy.squeeze(-1).numpy()
catalog['calculated_x'] = xy[:, 0]
catalog['calculated_y'] = xy[:, 1]
catalog = catalog[catalog['alt'] > 0].reset_index(drop=True)
draw_catalog(asd.bgr_data * 255, catalog, 'catalog.png')

# match stars
matcher = StarMatcher(10)   # the tolerance of matching is 10 pixels
pair = matcher.match(catalog, extracted_stars)
pair = pair[pair['extracted_x'].notna()]  # remove the unpaired rows.
print(f'match {len(pair)} points')

# Training the model
cali = Calibrate(_image_size=image_shape, model='AltAzToPhotoPoly')
loss = cali.train(pair["alt"].values, pair["az"].values,
                  pair["extracted_x"].values, pair["extracted_y"].values)

# Save model
cali.save('calibration.pt')

# show the center and radius in different altitudes
print(f'center: {cali.center}')
altitudes = [10, 20, 30, 40, 50, 60, 70, 80]
for alt in altitudes:
    print(f'{alt} | {cali.radius(alt)}')

print(f'north_theta: {cali.north_theta}')

```


### Visibility

After you get the camera model, you can calculate the visibility of the stars. The following codes show how to calculate the visibility of the stars in the Tycho-2 catalog.

Note: codes is incomplete. You need to fill in the location of the allsky camera, the image shape, the path of the image you want to process, and the limit magnitude of the camera by your judgement.

```python
import pandas as pd
import sqlalchemy
from astropy.coordinates import EarthLocation
from astropy_healpix import HEALPix
import astropy.units as u

from allskyia import Allsky, Calibrate, plot_hp_visibility

# Loading catalog. Here we use Tycho-2 catalog.
# `catalog_full` is a pandas DataFrame object. The column names keep the same as the original catalog.
tycho_engin = sqlalchemy.create_engine('DATABASE URL')
tycho_engin.connect()
catalog_full = pd.read_sql_query(f"SELECT * FROM TABLENAME ORDER BY VTmag", tycho_engin)

# Fill in the location of the allsky camera. 
# You can use EarthLocation.from_site('SITE NAME') as well
location = EarthLocation.from_geodetic(lon=, lat=, height=)

# Image shape and HEALPix object
image_shape = (rows, cols)  # Replace with the actual image shape
hp = HEALPix(nside=16, order='ring')

# Camera calibration model
camera_model = Calibrate(image_shape, 'AltAzToPhotoPoly')
camera_model.load('calibration.pt')

# Create Allsky object
catalog = catalog_full[catalog_full['VTmag'] < MAX_MAG]  # Replace MAX_MAG with the limit magnitude camera can observe.
allsky = Allsky(location=location, catalog=catalog, image_shape=image_shape)
allsky.set_calibrate_model(camera_model)

# Replace IMAGE PATH with the path of the image you want to process. 
# Both .fits and .jpeg are supported.
allsky.load_data('IMAGE PATH')  
v = allsky.visibility(hp, horizon=20 * u.deg)    # Replace with the horizon you want to set
plot_hp_visibility(v, hp, theta_offset=allsky.north_theta_rad, save_path=None)  # Replace with the path you want to save the plot

```

