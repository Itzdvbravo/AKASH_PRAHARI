import { describe, it, expect } from 'vitest';
import { searchTiles } from '../search.api';
import { fetchComparison, detectChange } from '../comparison.api';
import { fetchHealth } from '../system.api';

describe('TerraEyes Frontend API Subsystem', () => {
  it('should retrieve mock search tiles when query is executed', async () => {
    const res = await searchTiles({
      query_type: 'text',
      query_text: 'paris',
      filters: { sensor: 'sentinel-2', location: 'paris' }
    });

    expect(res).toBeDefined();
    expect(res.results.length).toBeGreaterThan(0);
    expect(res.results[0].tile_ref.location_id).toBe('paris');
    expect(res.results[0].confidence.score).toBeGreaterThan(0.8);
  });

  it('should fetch comparison pair data', async () => {
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
  });

  it('should execute change detection and return bounding boxes', async () => {
    const det = await detectChange({
      location_id: 'paris',
      tile_id: 'paris_2018_03_10'
    });

    expect(det.status).toBe('completed');
    expect(det.mask_url).toBeDefined();
    expect(det.bounding_boxes.length).toBeGreaterThan(0);
    expect(det.summary.change_type).toContain('Urban');
  });

  it('should return system diagnostics in mock mode', async () => {
    const health = await fetchHealth();
    expect(health.status).toBe('healthy');
    expect(health.embedding_model).toBeDefined();
    expect(health.change_detector).toBeDefined();
  });
});
