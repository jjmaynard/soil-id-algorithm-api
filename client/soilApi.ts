/**
 * Soil ID Algorithm API Client
 * 
 * TypeScript client library for interacting with the Soil ID Algorithm API
 * Copy this file to your project and customize as needed.
 * 
 * Usage:
 *   import { soilApi } from '@/lib/soilApi';
 *   const result = await soilApi.analyzeSoil(location, measurements);
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_SOIL_API_URL || 'http://localhost:8000';

// ============================================================================
// Types
// ============================================================================

export interface Location {
  lon: number;
  lat: number;
}

export interface FieldMeasurements {
  soilHorizon?: (string | null)[];
  topDepth?: (number | null)[];
  bottomDepth?: (number | null)[];
  rfvDepth?: (string | null)[];
  lab_Color?: (number[] | null)[];
  pSlope?: number | null;
  pElev?: number | null;
  bedrock?: number | null;
  cracks?: boolean | null;
}

export interface SoilListData {
  soil_list_json: {
    metadata: {
      location: string;
      model: string;
      unit_measure: Record<string, string>;
    };
    AWS_PIW90: any;
    'Soil Data Value': any;
    soilList: any[];
  };
  rank_data_csv: string;
  map_unit_component_data_csv: string;
}

export interface RankingResult {
  metadata: {
    location: string;
    model: string;
    unit_measure: Record<string, string>;
  };
  soilList: any[];
}

export interface ApiError {
  detail: string;
}

export interface HealthCheckResponse {
  status: string;
  message: string;
  version: string;
}

// ============================================================================
// Client Class
// ============================================================================

export class SoilApiClient {
  private baseUrl: string;
  private defaultHeaders: HeadersInit;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/$/, ''); // Remove trailing slash
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }

  /**
   * Make a request to the API
   */
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const response = await fetch(url, {
      ...options,
      headers: {
        ...this.defaultHeaders,
        ...options.headers,
      },
    });

    if (!response.ok) {
      let errorMessage = `API error: ${response.status}`;
      try {
        const errorData: ApiError = await response.json();
        errorMessage = errorData.detail || errorMessage;
      } catch {
        // If response is not JSON, use status text
        errorMessage = response.statusText || errorMessage;
      }
      throw new Error(errorMessage);
    }

    return response.json();
  }

  /**
   * Check API health status
   */
  async healthCheck(): Promise<HealthCheckResponse> {
    return this.request<HealthCheckResponse>('/');
  }

  /**
   * Get soil components at a location (Step 1 of two-step workflow)
   * 
   * @param location - Geographic coordinates
   * @returns Soil list data that can be stored and used later for ranking
   * 
   * @example
   * const soilList = await client.listSoils({ lon: -101.97, lat: 33.81 });
   * // Store soilList for later use
   * localStorage.setItem('soilList', JSON.stringify(soilList));
   */
  async listSoils(location: Location): Promise<SoilListData> {
    return this.request<SoilListData>('/api/list-soils', {
      method: 'POST',
      body: JSON.stringify(location),
    });
  }

  /**
   * Rank soils with field measurements (Step 2 of two-step workflow)
   * 
   * @param location - Geographic coordinates (same as used in listSoils)
   * @param soilListData - Data returned from listSoils()
   * @param fieldMeasurements - Field measurements collected from the site
   * @returns Ranked soil components based on similarity to measurements
   * 
   * @example
   * const soilList = JSON.parse(localStorage.getItem('soilList'));
   * const ranking = await client.rankSoils(
   *   { lon: -101.97, lat: 33.81 },
   *   soilList,
   *   { soilHorizon: ['Sandy loam'], topDepth: [0], bottomDepth: [20] }
   * );
   */
  async rankSoils(
    location: Location,
    soilListData: SoilListData,
    fieldMeasurements: FieldMeasurements
  ): Promise<RankingResult> {
    return this.request<RankingResult>('/api/rank-soils', {
      method: 'POST',
      body: JSON.stringify({
        ...location,
        ...soilListData,
        ...fieldMeasurements,
      }),
    });
  }

  /**
   * Combined operation: List and rank soils in one call (Recommended)
   * 
   * This is simpler and more efficient for most use cases.
   * Use the two-step workflow only if you need to show the soil list
   * to users before collecting field measurements.
   * 
   * @param location - Geographic coordinates
   * @param fieldMeasurements - Field measurements collected from the site
   * @returns Ranked soil components
   * 
   * @example
   * const result = await client.analyzeSoil(
   *   { lon: -101.97, lat: 33.81 },
   *   {
   *     soilHorizon: ['Sandy loam', 'Clay loam'],
   *     topDepth: [0, 20],
   *     bottomDepth: [20, 50],
   *     rfvDepth: ['0-1%', '1-15%'],
   *     lab_Color: [[50.5, 5.2, 20.1], [45.3, 6.1, 18.5]],
   *     pSlope: 5.0,
   *     pElev: 800.0,
   *   }
   * );
   */
  async analyzeSoil(
    location: Location,
    fieldMeasurements: FieldMeasurements
  ): Promise<RankingResult> {
    return this.request<RankingResult>('/api/analyze-soil', {
      method: 'POST',
      body: JSON.stringify({
        ...location,
        ...fieldMeasurements,
      }),
    });
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

/**
 * Default client instance
 * Uses NEXT_PUBLIC_SOIL_API_URL environment variable or localhost
 */
export const soilApi = new SoilApiClient();

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Validate location coordinates
 */
export function validateLocation(location: Location): boolean {
  return (
    typeof location.lon === 'number' &&
    typeof location.lat === 'number' &&
    location.lon >= -180 &&
    location.lon <= 180 &&
    location.lat >= -90 &&
    location.lat <= 90
  );
}

/**
 * Validate field measurements structure
 */
export function validateFieldMeasurements(measurements: FieldMeasurements): boolean {
  // Check if arrays have consistent lengths
  const arrays = [
    measurements.soilHorizon,
    measurements.topDepth,
    measurements.bottomDepth,
    measurements.rfvDepth,
    measurements.lab_Color,
  ].filter(arr => arr !== undefined);

  if (arrays.length === 0) return true; // No measurements is valid

  const lengths = arrays.map(arr => arr?.length || 0);
  const allSameLength = lengths.every(len => len === lengths[0]);

  return allSameLength;
}

/**
 * Create a field measurements object with proper structure
 */
export function createFieldMeasurements(
  horizons: Array<{
    soilHorizon?: string;
    topDepth?: number;
    bottomDepth?: number;
    rfvDepth?: string;
    lab_Color?: [number, number, number];
  }>,
  siteData?: {
    pSlope?: number;
    pElev?: number;
    bedrock?: number;
    cracks?: boolean;
  }
): FieldMeasurements {
  return {
    soilHorizon: horizons.map(h => h.soilHorizon || null),
    topDepth: horizons.map(h => h.topDepth ?? null),
    bottomDepth: horizons.map(h => h.bottomDepth ?? null),
    rfvDepth: horizons.map(h => h.rfvDepth || null),
    lab_Color: horizons.map(h => h.lab_Color || null),
    ...siteData,
  };
}
