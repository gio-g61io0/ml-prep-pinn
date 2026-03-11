


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


def classify_soil_texture(clay_pct, silt_pct, sand_pct):
    """Classify soil texture using the USDA soil texture triangle.

    Based on standard USDA-NRCS soil texture classification boundaries.

    Args:
        clay_pct: Clay percentage (0-100)
        silt_pct: Silt percentage (0-100)
        sand_pct: Sand percentage (0-100)

    Returns:
        Soil texture class string (e.g., 'Clay', 'Sandy Loam', etc.)
    """
    c, si, sa = clay_pct, silt_pct, sand_pct

    # --- High clay classes (check first) ---
    # if c >= 40 and si >= 40:
    #     return 'Silty Clay'
    # if c >= 35 and sa >= 45:
    #     return 'Sandy Clay'
    # if c >= 40:
    #     return 'Clay'

    # # --- Medium-high clay (27-40%) ---
    # if c >= 27 and c < 40 and sa < 20:
    #     return 'Silty Clay Loam'
    # if c >= 27 and c < 40:
    #     return 'Clay Loam'

    # # --- Sandy Clay Loam: 20-35% clay, <28% silt, >=45% sand ---
    # if c >= 20 and c < 35 and si < 28 and sa >= 45:
    #     return 'Sandy Clay Loam'

    # # --- High silt classes ---
    # if si >= 80 and c < 12:
    #     return 'Silt'
    # if si >= 50 and c < 27:
    #     return 'Silt Loam'

    # # --- Sandy classes ---
    # if sa >= 85 and c < 10:
    #     return 'Sand'
    # if sa >= 70 and sa < 90 and c < 15:
    #     return 'Loamy Sand'
    # if c < 20 and si < 50 and sa >= 43:
    #     return 'Sandy Loam'

    # # --- Loam: 7-27% clay, 28-50% silt, <=52% sand ---
    # if c >= 7 and c < 27 and si >= 28 and si < 50 and sa <= 52:
    #     return 'Loam'

    # # Fallback
    # if sa >= si and sa >= c:
    #     return 'Sandy Loam'
    # if si >= sa and si >= c:
    #     return 'Silt Loam'
    # return 'Loam'


    if abs((sa + si + c) - 100) > 0.1:
        return "Invalid Percentages"

    if c >= 40 and sa <= 45 and si <= 40:
        return "Clay"
    elif c >= 40 and sa > 45:
        return "Sandy Clay"
    elif c >= 40 and si > 40:
        return "Silty Clay"
    elif c >= 27 and c < 40 and sa > 45:
        return "Sandy Clay Loam"
    elif c >= 27 and c < 40 and sa <= 20 and si > 40:
        return "Silty Clay Loam"
    elif c >= 27 and c < 40 and sa <= 45 and sa > 20 and si <= 40:
        return "Clay Loam"
    elif c >= 12 and c < 27 and si < 28 and sa > 50:
        return "Sandy Clay Loam" # Overlap handling
    elif c >= 12 and c < 27 and si >= 28 and si < 50 and sa <= 52:
        return "Loam"

def gkg_to_percent(value_gkg):
    """Convert g/kg to percentage: (g/kg / 1000) * 100."""
    return (value_gkg / 1000.0) * 100.0


def add_soil_texture_column(df, clay_col='Clay_mean', silt_col='Silt_mean', sand_col='Sand_mean', new_col='soil_texture'):
    """Add a soil texture classification column to a DataFrame.

    Input columns are expected in g/kg. They are converted to percentages
    and classified using the USDA soil texture triangle.

    Args:
        df: DataFrame with clay, silt, sand columns in g/kg
        clay_col: Column name for clay (g/kg)
        silt_col: Column name for silt (g/kg)
        sand_col: Column name for sand (g/kg)
        new_col: Name of the new column to create

    Returns:
        DataFrame with the new soil texture column added
    """
    clay_pct = gkg_to_percent(df[clay_col].astype(float))
    silt_pct = gkg_to_percent(df[silt_col].astype(float))
    sand_pct = gkg_to_percent(df[sand_col].astype(float))

    df[new_col] = [
        classify_soil_texture(c, si, sa)
        for c, si, sa in zip(clay_pct, silt_pct, sand_pct)
    ]
    return df


SOIL_TEXTURE_TO_IDX = {
    'Sand': 0, 'Loamy Sand': 1, 'Sandy Loam': 2, 'Silt Loam': 3,
    'Loam': 4, 'Silt': 5, 'Sandy Clay Loam': 6, 'Clay Loam': 7,
    'Silty Clay Loam': 8, 'Sandy Clay': 9, 'Silty Clay': 10, 'Clay': 11,
}

def add_soil_texture_index(df, clay_col='Clay_mean', silt_col='Silt_mean', sand_col='Sand_mean'):
    """Classify soil texture from g/kg and add soil_texture + soil_texture_idx columns."""
    df = add_soil_texture_column(df, clay_col, silt_col, sand_col)
    df['soil_texture_idx'] = df['soil_texture'].map(SOIL_TEXTURE_TO_IDX).fillna(4).astype(int)
    return df


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