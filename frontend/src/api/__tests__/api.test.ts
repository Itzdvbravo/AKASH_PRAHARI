import { describe, it, expect, vi } from 'vitest';
import { apiClient } from '../client';
vi.mock('../client', () => ({
  apiClient: { get: vi.fn(), post: vi.fn() },
}));
import { searchTiles } from '../search.api';
import { fetchComparison, detectChange } from '../comparison.api';
import { fetchHealth } from '../system.api';

describe('TerraEyes Frontend API Subsystem', () => {
  it('sends search requests to the configured API and propagates failures', async () => {
    const error = new Error('API unavailable');
    vi.mocked(apiClient.post).mockRejectedValueOnce(error);
    await expect(searchTiles({ query_type: 'text', query_text: 'paris' })).rejects.toBe(error);
  });

  it('uses the backend comparison route', async () => {
    const post = vi.mocked(apiClient.post).mockResolvedValueOnce({ data: {
      comparison_id: 'cmp-live',
      before: { tile_ref: { tile_id: 'paris_tile', location_id: 'paris', date: '2016-03-15', sensor: 'sentinel-2' }, image_url: '/api/v1/images/paris_tile/2016-03-15', cloud_cover_pct: null },
      after: { tile_ref: { tile_id: 'paris_tile', location_id: 'paris', date: '2018-06-20', sensor: 'sentinel-2' }, image_url: '/api/v1/images/paris_tile/2018-06-20', cloud_cover_pct: null },
      coregistered: true,
    } });
    const comp = await fetchComparison({
      location_id: 'paris',
      tile_id: 'paris_2018_03_10',
      date_before: '2018-03-10',
      date_after: '2020-06-15',
      sensor: 'sentinel-2'
    });

    expect(comp.comparison_id).toBeDefined();
    expect(comp.before.image_url).toBeDefined();
    expect(comp.after.image_url).toBeDefined();
    expect(comp.coregistered).toBe(true);
    expect(post).toHaveBeenCalledWith('/api/v1/comparison', expect.any(Object));
  });

  it('uses the backend change detection route', async () => {
    const post = vi.mocked(apiClient.post).mockResolvedValueOnce({ data: { status: 'completed', mask_url: '/api/v1/masks/test.png', bounding_boxes: [], summary: { change_type: 'surface alteration' } } });
    await detectChange({ comparison_id: 'cmp-live' });
    expect(post).toHaveBeenCalledWith('/api/v1/change-detection', { comparison_id: 'cmp-live' });
  });

  it('propagates health check failures instead of presenting mock status', async () => {
    const error = new Error('API unavailable');
    vi.mocked(apiClient.get).mockRejectedValueOnce(error);
    await expect(fetchHealth()).rejects.toBe(error);
  });
});
