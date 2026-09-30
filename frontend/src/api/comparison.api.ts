import { apiClient } from './client';
import {
  ComparisonRequest,
  ComparisonResponse,
  ChangeDetectionRequest,
  ChangeDetectionResponse
} from '../types/api.types';
import { MOCK_COMPARISON_PARIS, MOCK_CHANGE_DETECTION_PARIS } from '../mock/mockData';

export async function fetchComparison(request: ComparisonRequest): Promise<ComparisonResponse> {
  try {
    const response = await apiClient.post<ComparisonResponse>('/api/v1/compare', request);
    return response.data;
  } catch {
    await new Promise((res) => setTimeout(res, 300));
    return MOCK_COMPARISON_PARIS;
  }
}

export async function detectChange(request: ChangeDetectionRequest): Promise<ChangeDetectionResponse> {
  try {
    const response = await apiClient.post<ChangeDetectionResponse>('/api/v1/change-detection', request);
    return response.data;
  } catch {
    await new Promise((res) => setTimeout(res, 350));
    return MOCK_CHANGE_DETECTION_PARIS;
  }
}
