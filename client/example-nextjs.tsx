/**
 * Example Next.js Component
 * 
 * This demonstrates how to use the Soil ID API in a Next.js application.
 * Copy and adapt this code for your use case.
 */

'use client';

import { useState } from 'react';
import { useSoilAnalysis } from './useSoilAnalysis';
import { createFieldMeasurements } from './soilApi';

// ============================================================================
// Example 1: Simple One-Step Analysis
// ============================================================================

export function SimpleSoilAnalysis() {
  const { analyze, loading, error, result } = useSoilAnalysis();
  const [location, setLocation] = useState({ lon: -101.9733687, lat: 33.81246789 });

  const handleAnalyze = async () => {
    await analyze(location, {
      soilHorizon: ['Sandy loam', 'Clay loam'],
      topDepth: [0, 20],
      bottomDepth: [20, 50],
      rfvDepth: ['0-1%', '1-15%'],
      lab_Color: [
        [50.5, 5.2, 20.1],
        [45.3, 6.1, 18.5],
      ],
      pSlope: 5.0,
      pElev: 800.0,
    });
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Soil Analysis</h2>

      <div className="mb-4">
        <label className="block mb-2">
          Longitude:
          <input
            type="number"
            value={location.lon}
            onChange={(e) => setLocation({ ...location, lon: Number(e.target.value) })}
            className="ml-2 px-2 py-1 border rounded"
            step="0.0001"
          />
        </label>
        <label className="block mb-2">
          Latitude:
          <input
            type="number"
            value={location.lat}
            onChange={(e) => setLocation({ ...location, lat: Number(e.target.value) })}
            className="ml-2 px-2 py-1 border rounded"
            step="0.0001"
          />
        </label>
      </div>

      <button
        onClick={handleAnalyze}
        disabled={loading}
        className="px-4 py-2 bg-blue-500 text-white rounded disabled:bg-gray-300"
      >
        {loading ? 'Analyzing...' : 'Analyze Soil'}
      </button>

      {error && (
        <div className="mt-4 p-4 bg-red-100 border border-red-400 rounded">
          <p className="text-red-700">Error: {error}</p>
        </div>
      )}

      {result && (
        <div className="mt-4">
          <h3 className="font-bold mb-2">Results:</h3>
          <div className="bg-gray-50 p-4 rounded overflow-auto max-h-96">
            <pre className="text-sm">{JSON.stringify(result, null, 2)}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Example 2: Form-Based Soil Analysis
// ============================================================================

interface HorizonFormData {
  soilHorizon: string;
  topDepth: number;
  bottomDepth: number;
  rfvDepth: string;
  lab_Color: [number, number, number];
}

export function SoilAnalysisForm() {
  const { analyze, loading, error, result, reset } = useSoilAnalysis();
  
  const [location, setLocation] = useState({ lon: -101.97, lat: 33.81 });
  const [horizons, setHorizons] = useState<HorizonFormData[]>([
    {
      soilHorizon: 'Sandy loam',
      topDepth: 0,
      bottomDepth: 20,
      rfvDepth: '0-1%',
      lab_Color: [50, 5, 20],
    },
  ]);
  const [siteData, setSiteData] = useState({
    pSlope: 5.0,
    pElev: 800.0,
    bedrock: null as number | null,
    cracks: false,
  });

  const addHorizon = () => {
    const lastHorizon = horizons[horizons.length - 1];
    setHorizons([
      ...horizons,
      {
        soilHorizon: '',
        topDepth: lastHorizon.bottomDepth,
        bottomDepth: lastHorizon.bottomDepth + 20,
        rfvDepth: '0-1%',
        lab_Color: [50, 5, 20],
      },
    ]);
  };

  const removeHorizon = (index: number) => {
    setHorizons(horizons.filter((_, i) => i !== index));
  };

  const updateHorizon = (index: number, field: keyof HorizonFormData, value: any) => {
    const updated = [...horizons];
    updated[index] = { ...updated[index], [field]: value };
    setHorizons(updated);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const measurements = createFieldMeasurements(horizons, siteData);
    await analyze(location, measurements);
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h2 className="text-2xl font-bold mb-6">Soil Analysis Form</h2>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Location Section */}
        <div className="bg-white p-4 rounded shadow">
          <h3 className="text-lg font-semibold mb-3">Location</h3>
          <div className="grid grid-cols-2 gap-4">
            <label className="block">
              <span className="text-sm font-medium">Longitude</span>
              <input
                type="number"
                value={location.lon}
                onChange={(e) => setLocation({ ...location, lon: Number(e.target.value) })}
                className="mt-1 block w-full px-3 py-2 border rounded"
                step="0.0001"
                required
              />
            </label>
            <label className="block">
              <span className="text-sm font-medium">Latitude</span>
              <input
                type="number"
                value={location.lat}
                onChange={(e) => setLocation({ ...location, lat: Number(e.target.value) })}
                className="mt-1 block w-full px-3 py-2 border rounded"
                step="0.0001"
                required
              />
            </label>
          </div>
        </div>

        {/* Horizons Section */}
        <div className="bg-white p-4 rounded shadow">
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-lg font-semibold">Soil Horizons</h3>
            <button
              type="button"
              onClick={addHorizon}
              className="px-3 py-1 bg-green-500 text-white rounded text-sm"
            >
              Add Horizon
            </button>
          </div>

          {horizons.map((horizon, index) => (
            <div key={index} className="mb-4 p-3 bg-gray-50 rounded">
              <div className="flex justify-between items-center mb-2">
                <span className="font-medium">Horizon {index + 1}</span>
                {horizons.length > 1 && (
                  <button
                    type="button"
                    onClick={() => removeHorizon(index)}
                    className="text-red-500 text-sm"
                  >
                    Remove
                  </button>
                )}
              </div>

              <div className="grid grid-cols-2 gap-3">
                <label className="block">
                  <span className="text-sm">Texture</span>
                  <input
                    type="text"
                    value={horizon.soilHorizon}
                    onChange={(e) => updateHorizon(index, 'soilHorizon', e.target.value)}
                    className="mt-1 block w-full px-2 py-1 border rounded text-sm"
                    placeholder="e.g., Sandy loam"
                  />
                </label>
                <label className="block">
                  <span className="text-sm">Rock Fragments</span>
                  <select
                    value={horizon.rfvDepth}
                    onChange={(e) => updateHorizon(index, 'rfvDepth', e.target.value)}
                    className="mt-1 block w-full px-2 py-1 border rounded text-sm"
                  >
                    <option>0-1%</option>
                    <option>1-15%</option>
                    <option>15-35%</option>
                    <option>35-60%</option>
                    <option>&gt;60%</option>
                  </select>
                </label>
                <label className="block">
                  <span className="text-sm">Top Depth (cm)</span>
                  <input
                    type="number"
                    value={horizon.topDepth}
                    onChange={(e) => updateHorizon(index, 'topDepth', Number(e.target.value))}
                    className="mt-1 block w-full px-2 py-1 border rounded text-sm"
                  />
                </label>
                <label className="block">
                  <span className="text-sm">Bottom Depth (cm)</span>
                  <input
                    type="number"
                    value={horizon.bottomDepth}
                    onChange={(e) => updateHorizon(index, 'bottomDepth', Number(e.target.value))}
                    className="mt-1 block w-full px-2 py-1 border rounded text-sm"
                  />
                </label>
              </div>
            </div>
          ))}
        </div>

        {/* Site Data Section */}
        <div className="bg-white p-4 rounded shadow">
          <h3 className="text-lg font-semibold mb-3">Site Data</h3>
          <div className="grid grid-cols-2 gap-4">
            <label className="block">
              <span className="text-sm font-medium">Slope (%)</span>
              <input
                type="number"
                value={siteData.pSlope || ''}
                onChange={(e) => setSiteData({ ...siteData, pSlope: Number(e.target.value) })}
                className="mt-1 block w-full px-3 py-2 border rounded"
                step="0.1"
              />
            </label>
            <label className="block">
              <span className="text-sm font-medium">Elevation (m)</span>
              <input
                type="number"
                value={siteData.pElev || ''}
                onChange={(e) => setSiteData({ ...siteData, pElev: Number(e.target.value) })}
                className="mt-1 block w-full px-3 py-2 border rounded"
                step="0.1"
              />
            </label>
            <label className="block">
              <span className="text-sm font-medium">Bedrock Depth (cm)</span>
              <input
                type="number"
                value={siteData.bedrock || ''}
                onChange={(e) =>
                  setSiteData({ ...siteData, bedrock: e.target.value ? Number(e.target.value) : null })
                }
                className="mt-1 block w-full px-3 py-2 border rounded"
                placeholder="Optional"
              />
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={siteData.cracks}
                onChange={(e) => setSiteData({ ...siteData, cracks: e.target.checked })}
                className="mr-2"
              />
              <span className="text-sm font-medium">Soil Cracks Present</span>
            </label>
          </div>
        </div>

        {/* Submit Button */}
        <div className="flex gap-3">
          <button
            type="submit"
            disabled={loading}
            className="px-6 py-2 bg-blue-500 text-white rounded disabled:bg-gray-300 font-medium"
          >
            {loading ? 'Analyzing...' : 'Analyze Soil'}
          </button>
          {result && (
            <button
              type="button"
              onClick={reset}
              className="px-6 py-2 bg-gray-500 text-white rounded"
            >
              Reset
            </button>
          )}
        </div>
      </form>

      {/* Error Display */}
      {error && (
        <div className="mt-6 p-4 bg-red-100 border border-red-400 rounded">
          <h4 className="font-bold text-red-800">Error</h4>
          <p className="text-red-700">{error}</p>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="mt-6 bg-white p-6 rounded shadow">
          <h3 className="text-xl font-bold mb-4">Analysis Results</h3>
          <div className="bg-gray-50 p-4 rounded overflow-auto max-h-96">
            <pre className="text-sm">{JSON.stringify(result, null, 2)}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Example 3: Using with React Query
// ============================================================================

/**
 * Example using @tanstack/react-query for caching and state management
 * 
 * Installation: npm install @tanstack/react-query
 * 
 * Setup in _app.tsx or layout.tsx:
 *   import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
 *   const queryClient = new QueryClient();
 *   <QueryClientProvider client={queryClient}>...</QueryClientProvider>
 */

/*
import { useQuery } from '@tanstack/react-query';
import { soilAnalysisQuery } from './useSoilAnalysis';

export function SoilAnalysisWithReactQuery() {
  const location = { lon: -101.97, lat: 33.81 };
  const measurements = {
    soilHorizon: ['Sandy loam'],
    topDepth: [0],
    bottomDepth: [20],
  };

  const { data, isLoading, error, refetch } = useQuery(
    soilAnalysisQuery(location, measurements)
  );

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div>
      <button onClick={() => refetch()}>Refresh</button>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}
*/
