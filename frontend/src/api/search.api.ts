import { apiClient } from './client';
import { SearchRequest, SearchResponse } from '../types/api.types';
import { getSearchResults, matchQueryToPreset } from '../mock/mockData';

export async function searchTiles(request: SearchRequest): Promise<SearchResponse> {
  try {
    const response = await apiClient.post<SearchResponse>('/api/v1/search', request);
    return response.data;
  } catch {
    // Graceful fallback to mock data when backend is not running
    await new Promise((res) => setTimeout(res, 250));

    const preset = matchQueryToPreset(request.query_text ?? '');
    const results = getSearchResults(preset.id);

    return {
      query_id: `mock_query_${Date.now()}`,
      results,
      total: results.length,
      query_ms: 42,
    };
  }
}
