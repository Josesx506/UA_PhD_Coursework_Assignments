import numpy as np
from numpy import sin,cos,tan,arcsin,arctan,degrees,radians


class Magnetic_Calc:
    def __init__(self,incl,decl,longitude,latitude) -> None:
        self.incl = radians(incl)
        self.decl = radians(decl)
        self.lon = longitude
        self.lat = radians(latitude)
    
    def calc_paleolatitude(self):
        paleolat = arctan(tan(self.incl)/2)
        self.paleolat = paleolat
        print(f"Paleolatitude: {degrees(paleolat):.2f}˚")

    def calc_paleomagnetic_poles(self):
        # Paleomagnetic latitude
        plm_lat = (sin(self.lat)*sin(self.paleolat)) + (cos(self.lat)*cos(self.paleolat)*cos(self.decl))
        plm_lat = degrees(arcsin(plm_lat))
        # Paleomagnetic longitude
        plm_lon = (cos(self.paleolat)*sin(self.decl)) / cos(radians(plm_lat))
        plm_lon = self.lon + degrees(arcsin(plm_lon))

        print(f"Paleomagnetic longitude: {plm_lon:.2f}˚")
        print(f"Paleomagnetic latitude: {plm_lat:.2f}˚")



t = Magnetic_Calc(68.5,23.5, 32, 50)
t.calc_paleolatitude()
t.calc_paleomagnetic_poles()