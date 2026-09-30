import { apiClient } from './client';
import { SearchRequest, SearchResponse } from '../types/api.types';
import { MOCK_SEARCH_RESULTS } from '../mock/mockData';

export async function searchTiles(request: SearchRequest): Promise<SearchResponse> {
  try {
    const response = await apiClient.post<SearchResponse>('/api/v1/search', request);
    return response.data;
  } catch {
    // Graceful fallback to mock data when backend is not running
    // Simulate short network delay for responsive UX feedback
    await new Promise((res) => setTimeout(res, 250));

    let filtered = [...MOCK_SEARCH_RESULTS];

    if (request.filters?.location && request.filters.location !== 'all') {
      const loc = request.filters.location.toLowerCase();
      filtered = filtered.filter(
        (item) => item.tile_ref.location_id.toLowerCase().includes(loc)
      );
    }

    if (request.filters?.sensor && request.filters.sensor !== 'any') {
      filtered = filtered.filter(
        (item) => item.tile_ref.sensor === request.filters?.sensor
      );
    }

    if (request.query_text && request.query_text.trim() !== '') {
      const q = request.query_text.toLowerCase();
      filtered = filtered.filter(
        (item) =>
          item.tile_ref.location_id.toLowerCase().includes(q) ||
          item.tile_ref.tile_id.toLowerCase().includes(q) ||
          q.includes('urban') ||
          q.includes('expansion') ||
          q.includes('change') ||
          q.includes('paris') ||
          q.includes('berlin')
      );
    }

    return {
      query_id: `mock_query_${Date.now()}`,
      results: filtered,
      total: filtered.length,
      query_ms: 42
    };
  }
}
