import { apiClient } from './client';
import { HealthResponse } from '../types/api.types';
import { MOCK_HEALTH } from '../mock/mockData';

export async function fetchHealth(): Promise<HealthResponse> {
  try {
    const response = await apiClient.get<HealthResponse>('/api/v1/health');
    return response.data;
  } catch {
    await new Promise((res) => setTimeout(res, 150));
    return MOCK_HEALTH;
  }
}
