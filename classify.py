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


df['classification'] = df['susceptibility'].apply(classify_susceptibility)

print(df['classification'].value_counts())
print(df['classification'].head())
print(df.columns)

df.to_file("/Users/giogonzales/Documents/ml-prep/mlprep/outputs/obj2/wetness/v3-Soil-Conditioned/geotechnical_values_classified.gpkg", driver="GPKG")


print(gpd.list_layers("/Users/giogonzales/Documents/ml-prep/mlprep/outputs/obj2/wetness/v3-Soil-Conditioned/geotechnical_values_classified.gpkg"))