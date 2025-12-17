# Client Integration Guide

## Calling the Soil ID API from Your Vercel Web App

This guide shows how to integrate the Soil ID Algorithm API into your frontend application.

## Quick Start

### 1. Environment Variables

In your web app, create a `.env.local` file:

```bash
# For local development (if API is running locally)
NEXT_PUBLIC_SOIL_API_URL=http://localhost:8000

# For production (replace with your deployed API URL)
NEXT_PUBLIC_SOIL_API_URL=https://your-soil-api.vercel.app
```

### 2. Installation

No special installation needed! The API uses standard HTTP requests.

For TypeScript projects, you can copy the types from `client/types.ts` (see below).

## Integration Examples

### Option 1: Combined Endpoint (Recommended for Most Cases)

This is the simplest approach - one API call does everything.

#### React/Next.js Example

```tsx
'use client'; // For Next.js App Router

import { useState } from 'react';

export default function SoilAnalysis() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const analyzeSoil = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_SOIL_API_URL}/api/analyze-soil`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          lon: -101.9733687,
          lat: 33.81246789,
          soilHorizon: ['Sandy loam', 'Clay loam'],
          topDepth: [0, 20],
          bottomDepth: [20, 50],
          rfvDepth: ['0-1%', '1-15%'],
          lab_Color: [[50.5, 5.2, 20.1], [45.3, 6.1, 18.5]],
          pSlope: 5.0,
          pElev: 800.0,
          bedrock: null,
          cracks: false,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <button onClick={analyzeSoil} disabled={loading}>
        {loading ? 'Analyzing...' : 'Analyze Soil'}
      </button>
      
      {error && <div className="error">Error: {error}</div>}
      
      {result && (
        <div>
          <h2>Results</h2>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
```

#### Vanilla JavaScript Example

```javascript
async function analyzeSoil(location, fieldData) {
  try {
    const response = await fetch('https://your-soil-api.vercel.app/api/analyze-soil', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        lon: location.lon,
        lat: location.lat,
        ...fieldData,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error analyzing soil:', error);
    throw error;
  }
}

// Usage
const result = await analyzeSoil(
  { lon: -101.97, lat: 33.81 },
  {
    soilHorizon: ['Sandy loam'],
    topDepth: [0],
    bottomDepth: [20],
    rfvDepth: ['0-1%'],
    lab_Color: [[50.5, 5.2, 20.1]],
    pSlope: 5.0,
  }
);
```

### Option 2: Separate Endpoints (For Two-Step Workflow)

Use this when you need to:
1. Get soil list first (show to user)
2. Later, collect field data and rank

#### React/Next.js Example with State Management

```tsx
'use client';

import { useState } from 'react';

interface SoilListData {
  soil_list_json: any;
  rank_data_csv: string;
  map_unit_component_data_csv: string;
}

export default function SoilWorkflow() {
  const [soilListData, setSoilListData] = useState<SoilListData | null>(null);
  const [rankResult, setRankResult] = useState(null);

  // Step 1: Get soil list
  const getSoilList = async (lon: number, lat: number) => {
    const response = await fetch(`${process.env.NEXT_PUBLIC_SOIL_API_URL}/api/list-soils`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lon, lat }),
    });

    const data = await response.json();
    setSoilListData(data); // Store in state
    
    // Optionally persist to localStorage
    localStorage.setItem('soilListData', JSON.stringify(data));
    
    return data;
  };

  // Step 2: Rank with field data
  const rankSoils = async (fieldData: any) => {
    if (!soilListData) {
      throw new Error('Must call getSoilList first');
    }

    const response = await fetch(`${process.env.NEXT_PUBLIC_SOIL_API_URL}/api/rank-soils`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        lon: fieldData.lon,
        lat: fieldData.lat,
        ...soilListData, // Include the stored data
        ...fieldData,
      }),
    });

    const result = await response.json();
    setRankResult(result);
    return result;
  };

  return (
    <div>
      {/* Step 1: Location Selection */}
      <button onClick={() => getSoilList(-101.97, 33.81)}>
        Get Soil List
      </button>

      {/* Step 2: After collecting field data */}
      {soilListData && (
        <button
          onClick={() =>
            rankSoils({
              lon: -101.97,
              lat: 33.81,
              soilHorizon: ['Sandy loam'],
              topDepth: [0],
              bottomDepth: [20],
              // ... other field measurements
            })
          }
        >
          Rank Soils
        </button>
      )}

      {rankResult && <pre>{JSON.stringify(rankResult, null, 2)}</pre>}
    </div>
  );
}
```

## Reusable Client Library

Create a `lib/soilApi.ts` file in your project:

```typescript
// lib/soilApi.ts

const API_BASE_URL = process.env.NEXT_PUBLIC_SOIL_API_URL || 'http://localhost:8000';

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
  soil_list_json: any;
  rank_data_csv: string;
  map_unit_component_data_csv: string;
}

class SoilApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Get soil components at a location
   */
  async listSoils(location: Location): Promise<SoilListData> {
    const response = await fetch(`${this.baseUrl}/api/list-soils`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(location),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to list soils');
    }

    return response.json();
  }

  /**
   * Rank soils with field measurements
   */
  async rankSoils(
    location: Location,
    soilListData: SoilListData,
    fieldMeasurements: FieldMeasurements
  ): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/rank-soils`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...location,
        ...soilListData,
        ...fieldMeasurements,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to rank soils');
    }

    return response.json();
  }

  /**
   * Combined: List and rank in one call
   */
  async analyzeSoil(
    location: Location,
    fieldMeasurements: FieldMeasurements
  ): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/analyze-soil`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...location,
        ...fieldMeasurements,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to analyze soil');
    }

    return response.json();
  }

  /**
   * Check API health
   */
  async healthCheck(): Promise<{ status: string; message: string }> {
    const response = await fetch(`${this.baseUrl}/`);
    return response.json();
  }
}

// Export singleton instance
export const soilApi = new SoilApiClient();

// Export class for custom instances
export { SoilApiClient };
```

### Using the Client Library

```tsx
// In your component
import { soilApi } from '@/lib/soilApi';

export default function SoilAnalysisPage() {
  const handleAnalysis = async () => {
    try {
      // Simple one-step analysis
      const result = await soilApi.analyzeSoil(
        { lon: -101.97, lat: 33.81 },
        {
          soilHorizon: ['Sandy loam'],
          topDepth: [0],
          bottomDepth: [20],
          pSlope: 5.0,
        }
      );
      console.log('Analysis result:', result);
    } catch (error) {
      console.error('Analysis failed:', error);
    }
  };

  const handleTwoStepWorkflow = async () => {
    try {
      // Step 1: Get soil list
      const soilList = await soilApi.listSoils({ lon: -101.97, lat: 33.81 });
      
      // Show soilList to user, collect measurements...
      
      // Step 2: Rank with measurements
      const ranking = await soilApi.rankSoils(
        { lon: -101.97, lat: 33.81 },
        soilList,
        { soilHorizon: ['Sandy loam'], topDepth: [0], bottomDepth: [20] }
      );
      console.log('Ranking result:', ranking);
    } catch (error) {
      console.error('Workflow failed:', error);
    }
  };

  return (
    <div>
      <button onClick={handleAnalysis}>Quick Analysis</button>
      <button onClick={handleTwoStepWorkflow}>Detailed Workflow</button>
    </div>
  );
}
```

## React Hook Example

For even cleaner code, create a custom hook:

```typescript
// hooks/useSoilAnalysis.ts

import { useState } from 'react';
import { soilApi, Location, FieldMeasurements } from '@/lib/soilApi';

export function useSoilAnalysis() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  const analyze = async (location: Location, measurements: FieldMeasurements) => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await soilApi.analyzeSoil(location, measurements);
      setResult(data);
      return data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setResult(null);
    setError(null);
  };

  return { analyze, loading, error, result, reset };
}

// Usage in component
function MyComponent() {
  const { analyze, loading, error, result } = useSoilAnalysis();

  const handleSubmit = async () => {
    await analyze(
      { lon: -101.97, lat: 33.81 },
      { soilHorizon: ['Sandy loam'], topDepth: [0], bottomDepth: [20] }
    );
  };

  return (
    <div>
      <button onClick={handleSubmit} disabled={loading}>
        {loading ? 'Analyzing...' : 'Analyze'}
      </button>
      {error && <p>Error: {error}</p>}
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}
```

## Next.js Server Actions (App Router)

For Next.js 13+, you can use Server Actions:

```typescript
// app/actions/soil.ts
'use server';

export async function analyzeSoil(location: { lon: number; lat: number }, fieldData: any) {
  const response = await fetch(`${process.env.SOIL_API_URL}/api/analyze-soil`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...location, ...fieldData }),
  });

  if (!response.ok) {
    throw new Error('Failed to analyze soil');
  }

  return response.json();
}

// In your component
'use client';

import { analyzeSoil } from './actions/soil';

export default function SoilForm() {
  const handleSubmit = async (formData: FormData) => {
    const result = await analyzeSoil(
      { lon: Number(formData.get('lon')), lat: Number(formData.get('lat')) },
      { /* field measurements */ }
    );
    console.log(result);
  };

  return <form action={handleSubmit}>...</form>;
}
```

## Error Handling Best Practices

```typescript
async function robustApiCall() {
  try {
    const response = await fetch(`${API_URL}/api/analyze-soil`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ /* data */ }),
      // Add timeout
      signal: AbortSignal.timeout(30000), // 30 second timeout
    });

    if (!response.ok) {
      // Handle specific status codes
      if (response.status === 404) {
        throw new Error('Soil data not available for this location');
      } else if (response.status === 500) {
        throw new Error('Server error processing soil data');
      }
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    if (error.name === 'AbortError') {
      console.error('Request timeout');
    }
    throw error;
  }
}
```

## Caching Strategies

### Client-Side Caching

```typescript
// Simple in-memory cache
const cache = new Map<string, { data: any; timestamp: number }>();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

async function cachedAnalyzeSoil(location: Location, measurements: FieldMeasurements) {
  const cacheKey = JSON.stringify({ location, measurements });
  const cached = cache.get(cacheKey);
  
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.data;
  }
  
  const data = await soilApi.analyzeSoil(location, measurements);
  cache.set(cacheKey, { data, timestamp: Date.now() });
  
  return data;
}
```

### React Query Integration

```typescript
import { useQuery } from '@tanstack/react-query';

function useSoilAnalysis(location: Location, measurements: FieldMeasurements) {
  return useQuery({
    queryKey: ['soil-analysis', location, measurements],
    queryFn: () => soilApi.analyzeSoil(location, measurements),
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: 2,
  });
}

// Usage
function Component() {
  const { data, isLoading, error } = useSoilAnalysis(
    { lon: -101.97, lat: 33.81 },
    { soilHorizon: ['Sandy loam'], topDepth: [0], bottomDepth: [20] }
  );
  
  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  return <pre>{JSON.stringify(data, null, 2)}</pre>;
}
```

## Performance Tips

1. **Use the combined endpoint** unless you specifically need the two-step workflow
2. **Implement caching** for repeated location queries
3. **Show loading states** - soil analysis can take several seconds
4. **Handle timeouts** - set appropriate timeout values
5. **Batch requests** if analyzing multiple locations
6. **Consider server-side rendering** for initial data in Next.js

## Security Considerations

1. **Environment Variables**: Use `NEXT_PUBLIC_` prefix only for client-side variables
2. **API Keys**: If you add authentication, store keys server-side only
3. **Rate Limiting**: Consider implementing rate limiting on your API
4. **CORS**: The API has CORS enabled - restrict origins in production

## Troubleshooting

### Common Issues

**CORS Errors:**
- Ensure `NEXT_PUBLIC_SOIL_API_URL` doesn't have trailing slash
- Check API CORS configuration in `api/main.py`

**Timeout Errors:**
- Increase timeout value in fetch options
- Soil analysis can take 5-10 seconds for complex calculations

**Type Errors:**
- Use the TypeScript types provided in `lib/soilApi.ts`
- Validate field measurements before sending

## Example Projects

See complete example implementations:
- **Next.js**: [examples/nextjs-client](examples/nextjs-client)
- **React**: [examples/react-client](examples/react-client)
- **Vue**: [examples/vue-client](examples/vue-client)
