import { apiClient } from './client';
import {
  ComparisonRequest,
  ComparisonResponse,
  ChangeDetectionRequest,
  ChangeDetectionResponse
} from '../types/api.types';

export async function fetchComparison(request: ComparisonRequest): Promise<ComparisonResponse> {
  const response = await apiClient.post<ComparisonResponse>('/api/v1/comparison', request);
  return response.data;
}

export async function detectChange(request: ChangeDetectionRequest): Promise<ChangeDetectionResponse> {
  const response = await apiClient.post<ChangeDetectionResponse>('/api/v1/change-detection', request);
  return response.data;
}
