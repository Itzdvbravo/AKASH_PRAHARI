import { apiClient } from './client';
import {
  ComparisonRequest,
  ComparisonResponse,
  ChangeDetectionRequest,
  ChangeDetectionResponse
} from '../types/api.types';
import {
  LOCATIONS,
  DEFAULT_LOCATION,
  buildComparison,
  buildChangeDetection
} from '../mock/mockData';

export async function fetchComparison(request: ComparisonRequest): Promise<ComparisonResponse> {
  try {
    const response = await apiClient.post<ComparisonResponse>('/api/v1/compare', request);
    return response.data;
  } catch {
    await new Promise((res) => setTimeout(res, 300));
    const loc = LOCATIONS.find(l => l.id === request.location_id) || DEFAULT_LOCATION;
    return buildComparison(loc);
  }
}

export async function detectChange(request: ChangeDetectionRequest): Promise<ChangeDetectionResponse> {
  try {
    const response = await apiClient.post<ChangeDetectionResponse>('/api/v1/change-detection', request);
    return response.data;
  } catch {
    await new Promise((res) => setTimeout(res, 350));
    const loc = LOCATIONS.find(l => l.id === request.location_id) || DEFAULT_LOCATION;
    return buildChangeDetection(loc);
  }
}

