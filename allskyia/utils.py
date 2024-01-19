import cv2 as cv
import numpy as np
import torch
from astropy_healpix import HEALPix
from astropy_healpix.healpy import ang2pix
from matplotlib import pyplot as plt
from photutils.aperture import CircularAperture


def draw_stars(data, sources, save_path='stars.png'):
    """
    Function to draw stars on a matplotlib figure and save the figure.

    Parameters:
    data (numpy.ndarray): The image data to be plotted.
    sources (list): List of sources to be plotted.
    save_path (str, optional): The path where the figure will be saved. Defaults to 'stars.png'.

    Returns:
    None
    """
    positions = np.transpose((sources['extracted_x'], sources['extracted_y']))

    for i, row in sources.iterrows():
        center = (int(row['extracted_x']), int(row['extracted_y']))

        cv.circle(data, center, 5, (200, 0, 200), 1)
        cv.ellipse(data,
                   center,
                   (int(3 * row['a']), int(3 * row['b'])),
                   int(row['theta'] * 180. / np.pi),
                   0, 360, (255, 0, 255), 1)

    if save_path is not None:
        cv.imwrite(save_path, data)

    return data


def draw_catalog(data, catalog, save_path='stars.png'):
    """
    Function to draw stars on a matplotlib figure and save the figure.

    Parameters:
    data (numpy.ndarray): The image data to be plotted.
    sources (list): List of sources to be plotted.
    save_path (str, optional): The path where the figure will be saved. Defaults to 'stars.png'.

    Returns:
    None
    """

    mask_image = np.zeros_like(data)
    for _, pair in catalog.iterrows():
        point = (int(pair['calculated_x']), int(pair['calculated_y']))
        radius = int(9 - pair['VTmag'])

        cv.circle(mask_image, point, radius, (173, 255, 47), -1)  # ADFF2F yellowgreen

    result = 0.5 * mask_image + data
    cv.imwrite(save_path, result)


def altaz_to_photo(alt, az, A, B):
    """
    Function to convert altitude and azimuth to photo coordinates.

    Parameters:
    alt (float): The altitude to be converted. unit: degree
    az (float): The azimuth to be converted. unit: degree
    A (torch.Tensor, optional): The transformation tensor for altitude and azimuth. Defaults to _A.
    [[-1 / 90, 0, 1],
     [0, 1, north_theta],
     [0, 0, 1]]
    B (torch.Tensor, optional): The transformation tensor for photo coordinates. Defaults to _B.
    [[s, 0, x0],
     [0, s, y0]]

    Returns:
    torch.Tensor: The photo coordinates.
    """
    if not isinstance(alt, torch.Tensor):
        alt = torch.tensor(alt, dtype=torch.float)

    if not isinstance(az, torch.Tensor):
        az = torch.tensor(az, dtype=torch.float)

    x = torch.stack([alt, az, torch.ones_like(az)], dim=1)
    x = x.unsqueeze(-1)
    points_polar = torch.matmul(A, x)

    points_polar[:, 1] = points_polar[:, 1] % 360

    rho = points_polar[:, 0]
    theta = points_polar[:, 1] / 180 * torch.pi
    points_cartesian = torch.stack([rho * torch.cos(theta),
                                    rho * torch.sin(theta),
                                    torch.ones_like(rho)], dim=1)

    result = torch.matmul(B, points_cartesian)

    return result


def draw_pairs(data, pairs, save_path='pairs.png'):
    """
    Function to draw pairs on a matplotlib figure and save the figure.

    Parameters:
    data (numpy.ndarray): The image data to be plotted.
    pairs (list): pd.DataFrame of pairs to be plotted.
    save_path (str, optional): The path where the figure will be saved. Defaults to 'pairs.png'.

    Returns:
    None
    """
    # The following line of code is used to clean the DataFrame 'pairs'.
    # It checks the column 'extracted_x' for any NaN (Not a Number) values.
    # If a row in the 'extracted_x' column contains a NaN value, that entire row is dropped from the DataFrame.
    # The 'dropna' function is used with the 'subset' parameter to specify the column to check for NaN values.
    # The result is a DataFrame 'pairs' that no longer contains any rows with NaN values in the 'extracted_x' column.
    pairs = pairs.dropna(subset=['extracted_x'])

    for _, pair in pairs.iterrows():
        start_point = (int(pair['extracted_x']), int(pair['extracted_y']))
        end_point = (int(pair['calculated_x']), int(pair['calculated_y']))

        cv.line(data,
                start_point,
                end_point,
                (50, 255, 255),
                2)

        cv.circle(data, start_point, 5, (255, 0, 255), 2)

    cv.imwrite(save_path, data)

    return


def alt_to_airmass(alt):
    """
    Function to calculate airmass.

    Parameters:
    alt (float): The altitude to be converted.

    Returns:
    float: The airmass.
    """
    return 1 / np.cos(np.deg2rad(90 - alt))


def plot_healpix(hp_data, hp: HEALPix, theta_offset, cmap_name, title,
                 save_path):
    # plot
    mesh_size = int(hp.nside * 100)
    theta = np.linspace(0, np.pi / 2, mesh_size)
    phi = np.linspace(0, 2 * np.pi, mesh_size)
    PHI, THETA = np.meshgrid(phi, theta)

    grid_pix = ang2pix(hp.nside, THETA, PHI)
    grid_map = hp_data.to_numpy().flatten()[grid_pix]

    longitude = phi
    latitude = np.rad2deg(theta)

    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    ax.set_theta_direction('clockwise')
    ax.set_theta_offset(theta_offset)
    plt.sca(ax)

    cmap = plt.get_cmap(cmap_name)
    ax.set_facecolor('orange')

    ret = plt.contourf(longitude,
                       latitude,
                       grid_map,
                       # vmin=0,
                       # vmax=1,
                       levels=50,
                       cmap=cmap, )

    plt.colorbar(ret, pad=0.1, ticks=np.linspace(0, 1, 11), extend='both')
    plt.title(title)

    if save_path is not None:
        # change suffix to png
        if save_path.endswith('.eps'):
            plt.savefig(save_path)  # save eps
            save_path = save_path.replace('.eps', '.png')
        elif save_path.endswith('.svg'):
            plt.savefig(save_path)  # save svg
            save_path = save_path.replace('.svg', '.png')
        plt.savefig(save_path, dpi=300)  # save png
    plt.show()


def plot_hp_extinction(hp_data, hp: HEALPix, cmap_name='Blues_r', title='Extinction',
                       save_path='extinction_equidistant.png', theta_offset=-2.715):
    plot_healpix(hp_data, hp, theta_offset, cmap_name=cmap_name, title=title, save_path=save_path)


def plot_hp_visibility(hp_data, hp: HEALPix, cmap_name='Blues', title='Visibility',
                       save_path='visibility_equidistant.png', theta_offset=-2.715):
    plot_healpix(hp_data, hp, theta_offset, cmap_name=cmap_name, title=title, save_path=save_path)
