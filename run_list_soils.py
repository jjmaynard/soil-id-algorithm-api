import json
from soil_id.us_soil import list_soils, rank_soils

# Run list_soils with the provided coordinates
# Note: list_soils expects (lon, lat, sim=True)
lon = -106.76431
lat = 32.25459

print("Running list_soils...")
list_result = list_soils(lon, lat, sim=True)

# Write the list_soils result to a JSON file
# Extract the soil_list_json attribute from the SoilListOutputData object
with open('soil_list_output.json', 'w') as f:
    json.dump(list_result.soil_list_json, f, indent=2, default=str)
print("List soils result written to soil_list_output.json")

# Run rank_soils with no data entered
print("\nRunning rank_soils with no data...")
rank_result_no_data = rank_soils(
    lon=lon,
    lat=lat,
    list_output_data=list_result,
    soilHorizon=[None],
    topDepth=[None],
    bottomDepth=[None],
    rfvDepth=[None],
    lab_Color=[None],
    pSlope=None,
    pElev=None,
    bedrock=None,
    cracks=None
)

# Write the rank_soils (no data) result to a JSON file
with open('rank_soils_no_data.json', 'w') as f:
    json.dump(rank_result_no_data, f, indent=2, default=str)
print("Rank soils (no data) result written to rank_soils_no_data.json")

# Run rank_soils with clay texture in 0-10 cm depth
print("\nRunning rank_soils with clay texture in 0-10 cm depth...")
rank_result_with_clay = rank_soils(
    lon=lon,
    lat=lat,
    list_output_data=list_result,
    soilHorizon=["clay"],
    topDepth=[0],
    bottomDepth=[10],
    rfvDepth=[None],
    lab_Color=[None],
    pSlope=None,
    pElev=None,
    bedrock=None,
    cracks=None
)

# Write the rank_soils (with clay) result to a JSON file
with open('rank_soils_with_clay.json', 'w') as f:
    json.dump(rank_result_with_clay, f, indent=2, default=str)
print("Rank soils (with clay) result written to rank_soils_with_clay.json")

print("\nAll results written successfully!")
