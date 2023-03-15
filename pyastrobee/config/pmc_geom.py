import numpy as np

# From gnc/pmc/src/shared.cc
nozzle_widths = np.array([5.0, 5.0, 2.8, 2.8, 2.8, 2.8])
nozzle_offsets_1 = np.array(
    [
        [6.00, 4.01, -1.56],
        [-6.00, 4.01, 1.56],
        [2.83, 6.00, 2.83],
        [-2.83, 6.00, -2.83],
        [-2.66, 4.01, 6.00],
        [2.66, 4.01, -6.00],
    ]
)
nozzle_offsets_2 = np.array(
    [
        [-6.00, -4.01, -1.56],
        [6.00, -4.01, 1.56],
        [-2.83, -6.00, 2.83],
        [2.83, -6.00, -2.83],
        [2.66, -4.01, 6.00],
        [-2.66, -4.01, -6.00],
    ]
)
nozzle_orientations_1 = np.array(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ]
)
nozzle_orientations_2 = np.array(
    [
        [-1, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ]
)
impeller_orientation_1 = np.array([0, 1, 0])
impeller_orientation_2 = np.array([0, -1, 0])
discharge_coeff_1 = np.array(
    [
        0.914971062,
        0.755778254,
        0.940762925,
        0.792109779,
        0.92401881,
        0.930319765,
    ]
)
discharge_coeff_2 = np.array(
    [
        0.947114008,
        0.764468916,
        1.000000000,
        0.90480943,
        0.936555627,
        0.893794766,
    ]
)
zero_thrust_area_1 = 0.0044667
zero_thrust_area_2 = 0.0042273

# This is an approximation, there is a whole lot of crazy calculations going on but the end result is a theta
# that is something like 0.000174532925, which is small enough that it shouldn't matter at all?
nozzle_thetas = np.zeros(6)  # NOTE THIS MAY ACTUALLY BE VERY INCORRECT???
# Other parameters from gnc/pmc/include/pmc/shared.h
units_inches_to_meters = 0.0254
abp_nozzle_min_open_angle = 15.68 * np.pi / 180.0
abp_nozzle_intake_height = 0.5154 * units_inches_to_meters
abp_nozzle_flap_length = 0.5353 * units_inches_to_meters

# CHECK ON THIS!! This seems weird. Doesn't the area need to depend on the commanded force?
nozzle_areas = (
    (
        abp_nozzle_intake_height
        - np.cos(abp_nozzle_min_open_angle + nozzle_thetas) * abp_nozzle_flap_length
    )
    * 2
    * nozzle_widths
)
air_density = 1.225  # or 1.2?
impeller_diameter = 5.5 * units_inches_to_meters
thrust_error_scale_factor = 1.25
# This impeller information is from A Brief Guide to Astrobee pg 57
quiet_impeller_speed = 209.4395
nominal_impeller_speed = 261.7994
aggressive_impeller_speed = 293.2153
