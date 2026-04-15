import json
from soil_id.global_soil import list_soils_global
from soil_id.db import get_datastore_connection

# Coordinates for the location (latitude, longitude)
# Note: list_soils_global expects (connection, lon, lat, buffer_dist)
lon = 36.35144
lat = 8.48144

print(f"Running list_soils_global at coordinates (lat: {lat}, lon: {lon})...")

# Establish database connection
try:
    connection = get_datastore_connection()
    print("Database connection established successfully.")
    
    # Run list_soils_global
    result = list_soils_global(connection, lon, lat)
    
    # Close the database connection
    connection.close()
    print("Database connection closed.")
    
    # Check if result is an error message string
    if isinstance(result, str):
        print(f"Result: {result}")
        output_data = {"status": "unavailable", "message": result}
    else:
        # Extract the soil_list_json from the SoilListOutputData object
        output_data = result.soil_list_json
    
    # Write the result to a JSON file
    with open('global_soil_list_output.json', 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print("Result written to global_soil_list_output.json")
    print(f"Result type: {type(result)}")
    
except Exception as e:
    print(f"Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
