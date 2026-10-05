import { apiClient } from './client';
import { HealthResponse } from '../types/api.types';

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>('/api/v1/health');
  return response.data;
}
