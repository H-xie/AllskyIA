class Photometry:

    def __init__(self, k, c, mag_zero):
        self.k = k
        self.c = c
        self.mag_zero = mag_zero

    def forward(self, catalog_mag, airmass):
        return self.k * airmass + self.c + catalog_mag

    def flux(self, catalog_mag, airmass):
        result = self.forward(catalog_mag, airmass)
        result = 10 ** ((result - self.mag_zero) / -2.5)
        return result
