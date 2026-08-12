import geopandas as gpd

def classify_susceptibility(value):
    if value >= 0.75:
        return 3
    if value < 0.75 and value >= 0.5:
        return 2
    # if value < 0.5 and value >= 0.25:
    #     return 1
    # else:
    #     return 1
    else:
        return 1
df = gpd.read_file("/Users/giogonzales/Documents/ml-prep/mlprep/outputs/obj2/wetness/v3-Soil-Conditioned/geotechnical_values.gpkg")
cotabato_data = gpd.read_file("/Users/giogonzales/Documents/ml-prep/mlprep/datasets/SU_19_v1_contri.gpkg")
cianjur_df = gpd.read_file("/Users/giogonzales/Documents/ml-prep/mlprep/datasets/cianjur_mapped.gpkg")


df.to_file("/Users/giogonzales/Documents/ml-prep/mlprep/outputs/obj2/wetness/v3-Soil-Conditioned/geotechnical_values_classified.gpkg", driver="GPKG")
cotabato_data.to_file("/Users/giogonzales/Documents/ml-prep/mlprep/datasets/SU_19_v1_contri-revised-contributing-factor.gpkg", driver="GPKG")

print(cianjur_df.columns)
cianjur_df['PGA2_max'] = cianjur_df['PGA2_max'] / 100
cianjur_df['SoilThc_mean'] = cianjur_df['SoilThc_mean'] / 100
cianjur_df['ContributingFactor_mean'] = cianjur_df['ContributingFactor_mean'] * 5
cianjur_df['BUK_mean'] = cianjur_df['BUK_mean'] * 0.0981

cianjur_df.to_file("/Users/giogonzales/Documents/ml-prep/mlprep/datasets/cianjur-mapped-transformed.gpkg", driver="GPKG")
