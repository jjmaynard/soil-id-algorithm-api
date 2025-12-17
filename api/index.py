# Copyright © 2024 Technology Matters
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from mangum import Mangum
import sys
from pathlib import Path

# Add the parent directory to the path to import soil_id
sys.path.insert(0, str(Path(__file__).parent.parent))

from soil_id.us_soil import list_soils, rank_soils, SoilListOutputData

app = FastAPI(
    title="Soil ID Algorithm API",
    description="API for soil identification and ranking",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class ListSoilsRequest(BaseModel):
    """Request model for listing soils at a location"""
    lon: float = Field(..., description="Longitude coordinate")
    lat: float = Field(..., description="Latitude coordinate")
    
    class Config:
        json_schema_extra = {
            "example": {
                "lon": -101.9733687,
                "lat": 33.81246789
            }
        }


class SoilListOutputDataResponse(BaseModel):
    """Response model that serializes SoilListOutputData"""
    soil_list_json: dict
    rank_data_csv: str
    map_unit_component_data_csv: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "soil_list_json": {"metadata": {}, "soilList": []},
                "rank_data_csv": "compname,sandpct_intpl,claypct_intpl...",
                "map_unit_component_data_csv": "mukey,cokey,compname..."
            }
        }


class RankSoilsRequest(BaseModel):
    """Request model for ranking soils with field data"""
    lon: float = Field(..., description="Longitude coordinate")
    lat: float = Field(..., description="Latitude coordinate")
    
    # SoilListOutputData fields (from previous list_soils call)
    soil_list_json: dict = Field(..., description="Soil list JSON from list_soils endpoint")
    rank_data_csv: str = Field(..., description="Rank data CSV from list_soils endpoint")
    map_unit_component_data_csv: str = Field(..., description="Map unit component data CSV from list_soils endpoint")
    
    # Field measurement data
    soilHorizon: Optional[List[Optional[str]]] = Field(None, description="Soil texture classifications")
    topDepth: Optional[List[Optional[float]]] = Field(None, description="Top depth of each horizon (cm)")
    bottomDepth: Optional[List[Optional[float]]] = Field(None, description="Bottom depth of each horizon (cm)")
    rfvDepth: Optional[List[Optional[str]]] = Field(None, description="Rock fragment volume classes")
    lab_Color: Optional[List[Optional[List[float]]]] = Field(None, description="LAB color values [L, A, B]")
    pSlope: Optional[float] = Field(None, description="Site slope percentage")
    pElev: Optional[float] = Field(None, description="Site elevation (m)")
    bedrock: Optional[float] = Field(None, description="Bedrock depth (cm)")
    cracks: Optional[bool] = Field(None, description="Presence of soil cracks")
    
    class Config:
        json_schema_extra = {
            "example": {
                "lon": -101.9733687,
                "lat": 33.81246789,
                "soil_list_json": {"metadata": {}, "soilList": []},
                "rank_data_csv": "compname,sandpct_intpl...",
                "map_unit_component_data_csv": "mukey,cokey...",
                "soilHorizon": ["Sandy loam", "Clay loam"],
                "topDepth": [0, 20],
                "bottomDepth": [20, 50],
                "rfvDepth": ["0-1%", "1-15%"],
                "lab_Color": [[50.5, 5.2, 20.1], [45.3, 6.1, 18.5]],
                "pSlope": 5.0,
                "pElev": 800.0,
                "bedrock": None,
                "cracks": False
            }
        }


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Soil ID Algorithm API",
        "version": "1.0.0"
    }


@app.get("/api/debug/environment")
async def debug_environment():
    """Debug endpoint to check installed packages and environment"""
    import sys
    import subprocess
    
    # Check if rosetta is importable
    rosetta_status = "not_installed"
    try:
        import rosetta
        rosetta_status = "imported_successfully"
    except ImportError as e:
        rosetta_status = f"import_failed: {str(e)}"
    
    # Check if serverless_soil_stats is available
    serverless_stats_status = "not_found"
    try:
        from soil_id.serverless_soil_stats import ilr, spearmanr
        serverless_stats_status = "imported_successfully"
    except ImportError as e:
        serverless_stats_status = f"import_failed: {str(e)}"
    
    # Get pip list
    try:
        pip_list = subprocess.check_output([sys.executable, "-m", "pip", "list"], 
                                          stderr=subprocess.STDOUT).decode()
        installed_packages = [line.split()[0] for line in pip_list.split('\n')[2:] if line.strip()]
    except Exception as e:
        installed_packages = [f"error: {str(e)}"]
    
    return {
        "python_version": sys.version,
        "rosetta_status": rosetta_status,
        "serverless_stats_status": serverless_stats_status,
        "installed_packages": installed_packages,
        "sys_path": sys.path[:5]  # First 5 paths
    }


@app.post("/api/list-soils", response_model=SoilListOutputDataResponse)
async def api_list_soils(request: ListSoilsRequest):
    """
    Get a list of soil components at a location.
    
    This endpoint queries soil databases (SSURGO/STATSGO) and returns
    soil component data that can be used for ranking with field measurements.
    
    The response should be stored client-side and passed to the rank-soils
    endpoint along with field measurement data.
    """
    try:
        result = list_soils(request.lon, request.lat)
        
        # Handle error case where result is a string
        if isinstance(result, str):
            raise HTTPException(status_code=404, detail=result)
        
        # Convert dataclass to dict for JSON serialization
        return SoilListOutputDataResponse(
            soil_list_json=result.soil_list_json,
            rank_data_csv=result.rank_data_csv,
            map_unit_component_data_csv=result.map_unit_component_data_csv
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving soil list: {str(e)}"
        )


@app.post("/api/rank-soils")
async def api_rank_soils(request: RankSoilsRequest):
    """
    Rank soil components based on field measurements.
    
    This endpoint requires the output from list-soils along with
    field measurement data to calculate similarity scores and rank
    the soil components.
    
    Workflow:
    1. Call list-soils to get soil component data
    2. Store the response client-side
    3. Collect field measurements
    4. Send both to this endpoint for ranking
    """
    try:
        # Reconstruct SoilListOutputData from request
        list_output_data = SoilListOutputData(
            soil_list_json=request.soil_list_json,
            rank_data_csv=request.rank_data_csv,
            map_unit_component_data_csv=request.map_unit_component_data_csv
        )
        
        # Call rank_soils with the reconstructed data
        result = rank_soils(
            lon=request.lon,
            lat=request.lat,
            list_output_data=list_output_data,
            soilHorizon=request.soilHorizon,
            topDepth=request.topDepth,
            bottomDepth=request.bottomDepth,
            rfvDepth=request.rfvDepth,
            lab_Color=request.lab_Color,
            pSlope=request.pSlope,
            pElev=request.pElev,
            bedrock=request.bedrock,
            cracks=request.cracks
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error ranking soils: {str(e)}"
        )


@app.post("/api/analyze-soil")
async def api_analyze_soil_combined(request: RankSoilsRequest):
    """
    Combined endpoint that performs both list and rank operations.
    
    This is a convenience endpoint that:
    1. Retrieves soil list for the location
    2. Immediately ranks them with provided field data
    
    Use this when you want to perform both operations in a single request.
    This is more efficient for Vercel serverless functions as it avoids
    the need to store intermediate data.
    """
    try:
        # First, get the soil list
        list_result = list_soils(request.lon, request.lat)
        
        # Handle error case
        if isinstance(list_result, str):
            raise HTTPException(status_code=404, detail=list_result)
        
        # Then rank with field data
        rank_result = rank_soils(
            lon=request.lon,
            lat=request.lat,
            list_output_data=list_result,
            soilHorizon=request.soilHorizon,
            topDepth=request.topDepth,
            bottomDepth=request.bottomDepth,
            rfvDepth=request.rfvDepth,
            lab_Color=request.lab_Color,
            pSlope=request.pSlope,
            pElev=request.pElev,
            bedrock=request.bedrock,
            cracks=request.cracks
        )
        
        # Combine list and rank results
        return {
            "soil_list_json": list_result.soil_list_json,
            "ranking_result": rank_result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing soil: {str(e)}"
        )


# For Vercel serverless deployment
app = app  # Vercel handles ASGI apps directly
