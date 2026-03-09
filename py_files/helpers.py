


import math
import re

def calculate_t(conductivity, soil_thickness):
    return conductivity * soil_thickness

def calculate_r(avg_rainfall_rate, infiltration_factors):
    return avg_rainfall_rate * infiltration_factors

def calculate_wetness_relative(slope_angle, r, t, contributing_factor):
    
    relative_wetness = (r * contributing_factor) / (t * math.sin(slope_angle))

    return min(relative_wetness, 1.0)  # Cap the value at 1.0

def convert_precipitation(precipitation):
    """
    Converts precipitation from mm/month -> mm/hr
    """
    return (precipitation * 0.001) / 720


SOIL_TYPE_TO_IDX = {
    'Sandy Clay Loam': 0,
    'Loam': 1,
    'Undifferentiated': 2,
}

def map_soil_type_to_conductivity(soil_type):
    """Maps a soil type string to its index for the HydraulicConductivityLayer."""
    return SOIL_TYPE_TO_IDX.get(soil_type, 2)  # default to Undifferentiated

def add_soil_type_index(df):
    """Creates a 'soil_type_idx' column from the 'type' column."""
    df['soil_type_idx'] = df['type'].map(SOIL_TYPE_TO_IDX).fillna(2).astype(int)
    return df