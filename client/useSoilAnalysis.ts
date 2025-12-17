/**
 * React Hook for Soil Analysis
 * 
 * Custom React hook that provides a clean interface for soil analysis
 * with loading, error, and result state management.
 * 
 * Usage:
 *   import { useSoilAnalysis } from '@/hooks/useSoilAnalysis';
 *   
 *   function MyComponent() {
 *     const { analyze, loading, error, result } = useSoilAnalysis();
 *     
 *     const handleClick = () => {
 *       analyze({ lon: -101.97, lat: 33.81 }, { soilHorizon: ['Sandy loam'] });
 *     };
 *     
 *     return <button onClick={handleClick}>Analyze</button>;
 *   }
 */

import { useState, useCallback } from 'react';
import { soilApi, Location, FieldMeasurements, RankingResult } from './soilApi';

interface UseSoilAnalysisReturn {
  analyze: (location: Location, measurements: FieldMeasurements) => Promise<RankingResult | null>;
  loading: boolean;
  error: string | null;
  result: RankingResult | null;
  reset: () => void;
}

export function useSoilAnalysis(): UseSoilAnalysisReturn {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RankingResult | null>(null);

  const analyze = useCallback(
    async (location: Location, measurements: FieldMeasurements): Promise<RankingResult | null> => {
      setLoading(true);
      setError(null);

      try {
        const data = await soilApi.analyzeSoil(location, measurements);
        setResult(data);
        return data;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
        setError(errorMessage);
        return null;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setLoading(false);
  }, []);

  return {
    analyze,
    loading,
    error,
    result,
    reset,
  };
}

// ============================================================================
// Two-Step Workflow Hook
// ============================================================================

interface UseSoilWorkflowReturn {
  listSoils: (location: Location) => Promise<void>;
  rankSoils: (location: Location, measurements: FieldMeasurements) => Promise<void>;
  soilList: any | null;
  ranking: RankingResult | null;
  loading: boolean;
  error: string | null;
  reset: () => void;
}

/**
 * Hook for two-step workflow: list soils, then rank with measurements
 */
export function useSoilWorkflow(): UseSoilWorkflowReturn {
  const [soilList, setSoilList] = useState<any>(null);
  const [ranking, setRanking] = useState<RankingResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const listSoils = useCallback(async (location: Location) => {
    setLoading(true);
    setError(null);

    try {
      const data = await soilApi.listSoils(location);
      setSoilList(data);
      
      // Optionally persist to localStorage
      if (typeof window !== 'undefined') {
        localStorage.setItem('soilListData', JSON.stringify(data));
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to list soils';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const rankSoils = useCallback(
    async (location: Location, measurements: FieldMeasurements) => {
      if (!soilList) {
        const error = 'Must call listSoils first';
        setError(error);
        throw new Error(error);
      }

      setLoading(true);
      setError(null);

      try {
        const data = await soilApi.rankSoils(location, soilList, measurements);
        setRanking(data);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to rank soils';
        setError(errorMessage);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [soilList]
  );

  const reset = useCallback(() => {
    setSoilList(null);
    setRanking(null);
    setError(null);
    setLoading(false);
    
    if (typeof window !== 'undefined') {
      localStorage.removeItem('soilListData');
    }
  }, []);

  return {
    listSoils,
    rankSoils,
    soilList,
    ranking,
    loading,
    error,
    reset,
  };
}

// ============================================================================
// React Query Integration
// ============================================================================

/**
 * For use with @tanstack/react-query
 * 
 * Installation: npm install @tanstack/react-query
 * 
 * Usage:
 *   import { useQuery } from '@tanstack/react-query';
 *   import { soilAnalysisQuery } from '@/hooks/useSoilAnalysis';
 *   
 *   function MyComponent() {
 *     const { data, isLoading, error } = useQuery(
 *       soilAnalysisQuery({ lon: -101.97, lat: 33.81 }, { soilHorizon: ['Sandy loam'] })
 *     );
 *   }
 */
export const soilAnalysisQuery = (location: Location, measurements: FieldMeasurements) => ({
  queryKey: ['soil-analysis', location, measurements],
  queryFn: () => soilApi.analyzeSoil(location, measurements),
  staleTime: 5 * 60 * 1000, // 5 minutes
  retry: 2,
  retryDelay: (attemptIndex: number) => Math.min(1000 * 2 ** attemptIndex, 30000),
});

export const soilListQuery = (location: Location) => ({
  queryKey: ['soil-list', location],
  queryFn: () => soilApi.listSoils(location),
  staleTime: 10 * 60 * 1000, // 10 minutes
  retry: 2,
});
