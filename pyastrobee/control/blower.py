# https://github.com/nasa/astrobee/blob/master/gnc/pmc/src/pmc_sim.cc

import numpy as np

from config import pmc_geom
from pyastrobee.control.physics_models import area_to_cdp_model


class Blower:
    def __init__(self, is_pmc1: bool):
        if is_pmc1:
            self.discharge_coeff = pmc_geom.discharge_coeff_1
            self.zero_thrust_area = pmc_geom.zero_thrust_area_1
            self.nozzle_offsets = pmc_geom.nozzle_offsets_1
            self.nozzle_orientations = pmc_geom.nozzle_orientations_1
            self.impeller_orientation = pmc_geom.impeller_orientation_1
        else:
            self.discharge_coeff = pmc_geom.discharge_coeff_2
            self.zero_thrust_area = pmc_geom.zero_thrust_area_2
            self.nozzle_offsets = pmc_geom.nozzle_offsets_2
            self.nozzle_orientations = pmc_geom.nozzle_orientations_2
            self.impeller_orientation = pmc_geom.impeller_orientation_2
        self.nozzle_areas = pmc_geom.nozzle_areas
        self.impeller_speed = pmc_geom.nominal_impeller_speed

    # Blower::Aerodynamics
    def thrust(self):
        area = self.zero_thrust_area + np.sum(self.discharge_coeff * self.nozzle_areas)
        cdp = area_to_cdp_model(area)
        delta_p = (
            self.impeller_speed**2
            * cdp
            * pmc_geom.impeller_diameter**2
            * pmc_geom.air_density
        )
        return (
            2
            * delta_p
            * self.discharge_coeff**2
            * self.nozzle_areas
            * pmc_geom.thrust_error_scale_factor
        )
