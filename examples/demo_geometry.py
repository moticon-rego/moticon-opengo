from moticon_opengo.geometry import sensor_areas_cm2
from moticon_opengo.text_export import Measurement, Side

# Force (N) per sensor is pressure (N/cm2) times sensor area (cm2). Gaps in
# between sensors are compensated by [area_factors].
force_per_sensor: np.array = \
    Measurement("export.txt").side_data(Side.LEFT).pressure * \
    sensor_areas_cm2 * area_factors
