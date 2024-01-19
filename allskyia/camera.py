"""
Solve camera calibration using PyTorch

The Projection step is referred to Kannala-Brandt model, which is a generalization of the
equidistant fisheye model. The model is described in the paper:
"Generic camera model and calibration method for conventional, wide-angle, and fisheye lenses"
by Juho Kannala and Sami S. Brandt
doi: https://doi.org/10.1109/TPAMI.2006.153

This code was implemented with reference to:
    1. Kannala-Brandt's well-documented Matlab code:
    https://users.aalto.fi/~kannalj1/calibration/calibration.html
    2. PyTorch Tutorial:
    https://pyorg/tutorials/beginner/basics/buildmodel_tutorial.html
    3. GitHub Copilot
"""
import time

import numpy as np
import torch


def projection(rho, k):
    """
    Kannala-Brandt general camera projection model

    The dimension of 'k' defines the order of the model.

    $$
    \rho = \sum_{i=1}^{n} k_i \rho^{2i+1}
    $$

    Parameters
    ----------
    rho :
        Radius in equidistant projection. Unit: rad
        It usually equals to `π - alt` in altitude-azimuth coordinate system
    k :
        Distortion coefficients
        n-dimensional array

    Returns
    -------
        radius of the projected point in camera coordinates

    """
    result = 0
    for ik in range(len(k)):
        order = 2 * ik + 1
        result = result + k[ik] * rho ** (order)

    return result


class AltAzToPhoto(torch.nn.Module):
    """
    Altitude-Azimuth to Photo coordinates

    Let $P = (x, y)$ be the photo coordinates,
    $A = (a, z)$ be the altitude-azimuth coordinates,

    Step1: Convert altitude-azimuth to equidistant projection. Consider rotation
    and translation of the camera, we have

    $$
    P_1 = R [\pi -a, z]^T + t
    $$
    where $R$ is the rotation matrix [4x4], $t$ is the translation vector [2].

    Step2: Project the point $P_1$ to the image plane. The projection function is
    defined as:

    $$
    \rho = s \times( \sum_{i=1}^{n} k_i \rho^{2i+1})
    $$
    where $s$ is the scale factor, $k$ is the distortion coefficients.

    Step3: Convert the polar coordinates to cartesian coordinates

    $$
    \begin{aligned}
    x = \rho \cos \theta + x_0
    y = \rho \sin \theta + y_0
    \end{aligned}
    $$

    """

    def __init__(self,
                 _R=None,
                 _t=None,
                 _scale=0.5,
                 _k=None,
                 _x0=0.5,
                 _y0=0.5):
        super().__init__()

        if _R is None:
            _R = [[1, 0],
                  [0, 1]]

        if _t is None:
            _t = torch.randn(2)

        if _k is None:
            _k = torch.randn(5)

        self.R = torch.nn.Parameter(torch.Tensor(_R))
        self.t = torch.nn.Parameter(_t)

        self.scale = torch.nn.Parameter(torch.Tensor([_scale]))
        self.k = torch.nn.Parameter(_k)  # KB投影参数

        self.x0 = torch.nn.Parameter(torch.Tensor([_x0]))
        self.y0 = torch.nn.Parameter(torch.Tensor([_y0]))

    def forward(self, alt, az):
        """
        Parameters
        ----------
        alt :
            altitude
        az :
            azimuth

        Returns
        -------
            photo coordinates

        """
        # step1: altaz to polar
        rho = (alt * self.R[0, 0] + az * self.R[0, 1]) + self.t[0]
        theta = alt * self.R[1, 0] + az * self.R[1, 1] + self.t[1]
        theta = theta % (2 * torch.pi)

        # step2: projection
        rho = self.scale * projection(rho, self.k)

        # step3: polar to cartesian
        x = rho * torch.cos(theta)
        y = rho * torch.sin(theta)

        x = x + self.x0
        y = y + self.y0

        return torch.stack([x, y], dim=1)

    def reset_parameters(self):

        self.R = torch.nn.Parameter(torch.Tensor([[1, 0],
                                                  [0, 1]]))
        self.t = torch.nn.Parameter(torch.randn(2))

        self.scale = torch.nn.Parameter(torch.Tensor([0.5]))
        self.k = torch.nn.Parameter(torch.randn(5))

        self.x0 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.y0 = torch.nn.Parameter(torch.Tensor([0.5]))


def polynomial(x, y, k):
    result_x = k[0] * x + k[1] * x ** 2 + k[2] * x ** 3 + k[3] * y + k[4] * y ** 2 + k[5] * y * x + k[6]
    result_y = k[7] * x + k[8] * x ** 2 + k[9] * x ** 3 + k[10] * y + k[11] * y ** 2 + k[12] * y * x + k[13]

    return result_x, result_y


def polynomial_full(x, y, k_x, k_y):
    assert k_x.dim() == 2
    assert k_x.shape[0] == k_x.shape[1]
    n_order = k_x.shape[0]

    X = torch.stack([x ** (i + 1) for i in range(n_order)], dim=1)
    Y = torch.stack([y ** (i + 1) for i in range(n_order)], dim=1)
    X = X.unsqueeze(-1)
    Y = Y.unsqueeze(1)

    result = torch.matmul(X, Y)
    result_x = torch.mul(result, k_x).sum(dim=[1, 2])
    result_y = torch.mul(result, k_y).sum(dim=[1, 2])

    return result_x, result_y


# train model 训练模型
class AltAzToPhotoPoly(torch.nn.Module):
    def __init__(self,
                 _R=None,
                 _t=None,
                 _scale=0.5,
                 _k=None,
                 _x0=0.5,
                 _y0=0.5,
                 _k_poly=None):
        super().__init__()
        if _R is None:
            _R = [[1, 0],
                  [0, 1]]

        if _t is None:
            _t = torch.randn(2)

        if _k is None:
            _k = torch.randn(5)

        if _k_poly is None:
            _k_poly = [0.144052, -0.000103, -0.000423,
                       -0.241509, -0.000279, 0.000014, 0.010287,
                       0.241530, 0.000047, -0.000032,
                       0.143713, -0.000366, -0.000058, 0.007811]

        self.R = torch.nn.Parameter(torch.Tensor(_R))
        self.t = torch.nn.Parameter(_t)

        self.scale = torch.nn.Parameter(torch.Tensor([_scale]))
        self.k = torch.nn.Parameter(_k)  # KB投影参数

        self.k_poly = torch.nn.Parameter(torch.Tensor(_k_poly))
        # self.k_poly_x = torch.nn.Parameter(torch.randn((5, 5)))
        # self.k_poly_y = torch.nn.Parameter(torch.randn((5, 5)))

        self.x0 = torch.nn.Parameter(torch.Tensor([_x0]))
        self.y0 = torch.nn.Parameter(torch.Tensor([_y0]))

    def forward(self, alt, az):
        # step1: altaz to polar
        rho = (alt * self.R[0, 0] + az * self.R[0, 1]) + self.t[0]
        theta = alt * self.R[1, 0] + az * self.R[1, 1] + self.t[1]

        # step2: projection
        rho = self.scale * projection(rho, self.k)

        # step3: polar to cartesian
        x = rho * torch.cos(theta)
        y = rho * torch.sin(theta)

        x, y = polynomial(x, y, self.k_poly)
        # x, y = polynomial_full(x, y, self.k_poly_x, self.k_poly_y)
        x = x + self.x0
        y = y + self.y0

        return torch.stack([x, y], dim=1)

    def reset_parameters(self):

        self.R = torch.nn.Parameter(torch.Tensor([[1, 0],
                                                  [0, 1]]))
        self.t = torch.nn.Parameter(torch.randn(2))

        self.scale = torch.nn.Parameter(torch.Tensor([0.5]))
        self.k = torch.nn.Parameter(torch.randn(5))

        self.k_poly = torch.nn.Parameter(torch.Tensor([0.144052, -0.000103, -0.000423,
                                                       -0.241509, -0.000279, 0.000014, 0.010287,
                                                       0.241530, 0.000047, -0.000032,
                                                       0.143713, -0.000366, -0.000058, 0.007811]))

        # self.k_poly_x = torch.nn.Parameter(torch.randn((5, 5)))
        # self.k_poly_y = torch.nn.Parameter(torch.randn((5, 5)))

        self.x0 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.y0 = torch.nn.Parameter(torch.Tensor([0.5]))


class Calibrate():
    def __init__(self,
                 _image_size: tuple | list | np.ndarray,
                 model="AltAzToPhoto",
                 _R=None,
                 _t=None,
                 _scale=0.5,
                 _k=None,
                 _x0=0.5,
                 _y0=0.5,
                 _lr=None,
                 _device="cpu"):

        self.is_loaded = None
        self.device = torch.device(_device)
        print(f"Using {self.device} device")

        self.is_print_log = True
        if _lr is None:
            _lr = 1e-3
        self.lr = _lr

        if model == "AltAzToPhoto":
            self.model = AltAzToPhoto(_R, _t, _scale, _k, _x0, _y0).to(self.device)
        elif model == "AltAzToPhotoPoly":
            self.model = AltAzToPhotoPoly(_R, _t, _scale, _k, _x0, _y0).to(self.device)
        self.criteria = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        assert len(_image_size) == 2
        self.image_size = np.array(_image_size)

    @property
    def valid_model(self):
        return ['AltAzToPhoto', 'AltAzToPhotoPoly']

    def prepare_data(self, alt, az, u, v):
        """
        Parameters
        ----------
        alt :
            altitude
        az :
            azimuth
        u :
            x coordinate in image
        v :
            y coordinate in image

        Returns
        -------
            input data for training

        """
        return self.prepare_altaz(alt, az), self.prepare_xy(u, v)

    def prepare_altaz(self, alt, az):
        # Construct input. Convert alt/az to rad
        rho = torch.deg2rad(90 - torch.Tensor(alt))
        theta = torch.deg2rad(torch.Tensor(az))

        return torch.stack([rho, theta], dim=1).to(self.device)

    def prepare_xy(self, u, v):
        # Construct output. Convert image coordinates to [0, 1]
        x = torch.Tensor(u) / self.image_size[0]
        y = torch.Tensor(v) / self.image_size[1]

        return torch.stack([x, y], dim=1).to(self.device)

    def train(self, alt, az, u, v, _epochs=0, outlier=5, stop_error=0.5):

        # # Construct input. Convert alt/az to rad
        # rho = torch.deg2rad(90 - torch.Tensor(alt))  # altitude 👉 radius. zenith is 0
        # theta = torch.deg2rad(torch.Tensor(az))
        # x = torch.stack([rho, theta], dim=1)
        #
        # # Construct output. Normalize u/v to [0, 1]
        # u = torch.Tensor(u) / self.image_size[0]
        # v = torch.Tensor(v) / self.image_size[1]
        # y = torch.stack([u, v], dim=1)

        x, y = self.prepare_data(alt, az, u, v)

        # decide epochs
        if _epochs == 0:
            # auto epochs. stop when loss < 1e-6
            epochs = int(1e6)
        else:
            # fixed epochs. only stop when epochs reached
            epochs = _epochs

        loss = None
        i = 0
        start_time = time.time()
        last_loss = 0
        while i < epochs:

            self.optimizer.zero_grad()

            # forward
            y_pred = self.model(x[..., 0], x[..., 1])

            # euclidean distance
            loss_full = torch.sum((y_pred - y) ** 2, dim=1)
            loss = loss_full.mean()

            if i == 0 and (loss.item() > 0.1 or torch.isnan(loss)):
                self.model.reset_parameters()
                self.model.to(self.device)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
                i = 0
                print(
                    f"epoch {i}, loss: {loss}, {np.sqrt(loss.item()) * self.image_size} pixel")
                continue

            if ((_epochs == 0 and i > 5e4 and loss.item() < 1e-8)
                    or (np.sqrt(loss.item()) * self.image_size.mean() < stop_error)
                    or torch.isnan(loss)):
                print(f"training finished at epoch {i}, because loss is {loss}")
                break

            # print training progress
            if self.is_print_log and i % 5000 == 0:
                print(
                    f"epoch {i}, loss: {loss}, {np.sqrt(loss.item()) * self.image_size} pixel, cost time: {time.time() - start_time} s")
                start_time = time.time()

                if abs(loss.item() - last_loss) < 1e-5 or (i == 0 and self.is_loaded):
                    np_loss_full = loss_full.detach().numpy()
                    loss_mean = loss.detach().numpy()
                    loss_std = np_loss_full.std()

                    # print outliers
                    threshold = loss_mean + outlier * loss_std
                    threshold = max(threshold, 1e-7)
                    print("threshold:", threshold)
                    index_remove = np.where(np_loss_full >= threshold)[0]
                    print("epoch", i, "outliers:", index_remove, "loss:", np_loss_full[index_remove],
                          "alt:", x[index_remove, 0], "az:", x[index_remove, 1],
                          "x:", y[index_remove, 0], "y:", y[index_remove, 1])

                    index_keep = np.where(np_loss_full < threshold)[0]
                    x = x[index_keep]
                    y = y[index_keep]

                    # loss after removing outliers
                    y_pred = self.model(x[..., 0], x[..., 1])
                    loss_full = torch.sum((y_pred - y) ** 2, dim=1)
                    loss = loss_full.mean()
                    print("remaining:", len(x), "loss:", loss.item())

                    if np.sqrt(loss.item()) * self.image_size.mean() < stop_error:
                        print(f"training finished at epoch {i}, because loss is {loss}")
                        break

                last_loss = loss.item()

            # backward
            loss.backward()
            self.optimizer.step()
            i += 1

        return loss.item()

    def save(self, path="calibration.pt"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="calibration.pt"):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self.is_loaded = True

    def directions(self):

        north_point = self.forward([0], [0]).flatten()
        center = self.forward([90], [0]).flatten()
        north = north_point - center
        radius = np.linalg.norm(north)

        point_alt20 = self.forward([20], [0]).flatten()
        radius_alt20 = np.linalg.norm(point_alt20 - center)

        north_theta = np.arctan(north[1] / north[0]) * 180 / np.pi
        if north[0] < 0:
            north_theta += 180
        elif north[1] < 0:
            north_theta += 360

        result = {'center': center,
                  'radius': radius,
                  'radius20': radius_alt20,
                  'north_theta': north_theta,
                  'north_point': north_point}

        return result

    def forward(self, alt, az) -> np.ndarray:
        """
        Parameters
        ----------
        alt :
            altitude
        az :
            azimuth

        Returns
        -------
            photo coordinates

        """
        x = self.prepare_altaz(alt, az)
        result = self.model(x[..., 0], x[..., 1])
        result = result.detach().numpy() * self.image_size

        return result

    @property
    def center(self):
        point_zenith = self.forward([90], [0]).flatten()
        return point_zenith

    def radius(self, alt):
        point = self.forward([alt], [0]).flatten()
        radius = np.linalg.norm(point - self.center)
        return radius

    @property
    def north_theta(self):
        north_point = self.forward([0], [0]).flatten()
        center = self.forward([90], [0]).flatten()
        north = north_point - center
        north_theta = np.arctan(north[1] / north[0]) * 180 / np.pi
        if north[0] < 0:
            north_theta += 180
        elif north[1] < 0:
            north_theta += 360
        return north_theta
