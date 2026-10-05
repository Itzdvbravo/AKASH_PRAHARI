// @vitest-environment jsdom
import { act } from 'react';
import { createRoot, Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App';
import { detectChange, fetchComparison } from '../api/comparison.api';
import { searchTiles } from '../api/search.api';
import { fetchHealth } from '../api/system.api';
import {
  ChangeDetectionResponse,
  ComparisonResponse,
  HealthResponse,
  SearchResponse,
} from '../types/api.types';

vi.mock('../api/comparison.api', () => ({
  detectChange: vi.fn(),
  fetchComparison: vi.fn(),
}));
vi.mock('../api/search.api', () => ({ searchTiles: vi.fn() }));
vi.mock('../api/system.api', () => ({ fetchHealth: vi.fn() }));

const scene: SearchResponse = {
  query_id: 'query-1',
  total: 1,
  query_ms: 12,
  results: [{
    rank: 1,
    tile_ref: {
      tile_id: 'beihai_0000_0000_sentinel-2',
      location_id: 'beihai',
      date: '2018-03-09',
      sensor: 'sentinel-2',
    },
    confidence: { score: 0.82, method: 'cosine_similarity', calibrated: false },
    thumbnail_url: '/api/v1/images/beihai_0000_0000_sentinel-2/2018-03-09/thumbnail',
    available_dates: ['2016-12-09', '2018-03-09'],
    geo_bbox: { west: 109.1, south: 21.45, east: 109.18, north: 21.52 },
  }],
};

const comparison: ComparisonResponse = {
  comparison_id: 'comparison-1',
  before: {
    tile_ref: { ...scene.results[0].tile_ref, date: '2016-12-09' },
    image_url: '/api/v1/images/beihai_0000_0000_sentinel-2/2016-12-09',
    cloud_cover_pct: null,
  },
  after: {
    tile_ref: scene.results[0].tile_ref,
    image_url: '/api/v1/images/beihai_0000_0000_sentinel-2/2018-03-09',
    cloud_cover_pct: null,
  },
  coregistered: true,
};

const changeDetection: ChangeDetectionResponse = {
  job_id: 'job-1',
  status: 'completed',
  mask_url: '/api/v1/masks/job-1.png',
  bounding_boxes: [],
  processing_ms: 42,
  summary: {
    location: 'beihai',
    date_before: '2016-12-09',
    date_after: '2018-03-09',
    change_type: 'structural_development',
    earliest_detectable_change: null,
    confidence: { score: 0.6, method: 'pixel_diff_otsu', calibrated: false },
    changed_pixel_fraction: 0.04,
    source_provenance: 'OSCD Sentinel-2',
    detector: 'pixel_diff_otsu',
  },
};

const health = (embeddingModel: string): HealthResponse => ({
  status: 'ok',
  version: '0.1.0',
  embedding_model: embeddingModel,
  change_detector: 'pixel_diff',
  faiss_index_size: 100,
  db_tiles: 200,
});

let container: HTMLDivElement;
let root: Root;

async function renderApp() {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => {
    root.render(<App />);
    await new Promise(resolve => setTimeout(resolve, 0));
  });
}

async function clickButton(label: string) {
  const button = Array.from(container.querySelectorAll('button'))
    .find(candidate => candidate.textContent?.includes(label));
  expect(button, `expected a button containing “${label}”`).toBeDefined();
  await act(async () => {
    button!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await new Promise(resolve => setTimeout(resolve, 0));
  });
}

async function inspectFirstResult() {
  const button = container.querySelector<HTMLButtonElement>('[id^="inspect-btn-"]');
  expect(button).not.toBeNull();
  await act(async () => {
    button!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await new Promise(resolve => setTimeout(resolve, 0));
  });
}

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  vi.mocked(searchTiles).mockResolvedValue(scene);
  vi.mocked(fetchHealth).mockResolvedValue(health('clip_vit_b32'));
  vi.mocked(fetchComparison).mockResolvedValue(comparison);
  vi.mocked(detectChange).mockResolvedValue(changeDetection);
});

afterEach(async () => {
  if (root) {
    await act(async () => root.unmount());
  }
  container?.remove();
  vi.clearAllMocks();
});

describe('dataset-backed application flow', () => {
  it('searches, opens a real comparison, and keeps curated progression out of dataset mode', async () => {
    await renderApp();
    expect(container.querySelectorAll('[id^="inspect-btn-"]').length).toBe(1);
    expect(vi.mocked(searchTiles)).toHaveBeenCalledWith(expect.objectContaining({
      query_type: 'text',
      top_k: 100,
    }));

    await inspectFirstResult();
    expect(vi.mocked(fetchComparison)).toHaveBeenCalledWith({
      location_id: 'beihai',
      tile_id: 'beihai_0000_0000_sentinel-2',
      date_before: '2016-12-09',
      date_after: '2018-03-09',
      sensor: 'sentinel-2',
    });
    expect(container.textContent).toContain('Analyst Change Detection Summary');
    expect(container.querySelectorAll('.comparison-layout img').length).toBeGreaterThan(0);
    expect(Array.from(container.querySelectorAll('button')).some(button =>
      button.textContent?.includes('View temporal progression')
    )).toBe(false);
  });

  it('reports the configured dataset and index in system health', async () => {
    await renderApp();
    await clickButton('System Health');

    expect(container.textContent).toContain('OSCD Sentinel-2 train split');
    expect(container.textContent).toContain('14 train cities indexed');
    expect(container.textContent).toContain('100 vectors');
    expect(container.textContent).toContain('200 scenes');
  });

  it('submits analyst text and reset searches through the search form', async () => {
    await renderApp();
    const input = container.querySelector<HTMLInputElement>('#nlp-search-input');
    expect(input).not.toBeNull();

    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      setter?.call(input, 'dense road grid near the coast');
      input!.dispatchEvent(new Event('input', { bubbles: true }));
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    await clickButton('Analyse');
    expect(vi.mocked(searchTiles)).toHaveBeenLastCalledWith(expect.objectContaining({
      query_text: 'dense road grid near the coast',
    }));

    await act(async () => {
      container.querySelector<HTMLButtonElement>('[aria-label="Clear search"]')!
        .dispatchEvent(new MouseEvent('click', { bubbles: true }));
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    expect(vi.mocked(searchTiles)).toHaveBeenLastCalledWith(expect.objectContaining({
      query_text: '',
      top_k: 100,
    }));
  });

  it('supports keyboard suggestions and closes the query menu on outside clicks', async () => {
    await renderApp();
    const input = container.querySelector<HTMLInputElement>('#nlp-search-input')!;
    await act(async () => {
      input.focus();
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    const firstSuggestion = container.querySelector('.search-dropdown-item .dropdown-query')?.textContent;
    expect(firstSuggestion).toBeTruthy();

    await act(async () => {
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    await act(async () => {
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    expect(container.querySelector('#search-dropdown')).toBeNull();
    expect(vi.mocked(searchTiles)).toHaveBeenLastCalledWith(expect.objectContaining({
      query_text: firstSuggestion,
    }));

    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      setter?.call(input, 'urban');
      input.dispatchEvent(new Event('input', { bubbles: true }));
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    expect(container.querySelector('#search-dropdown')).not.toBeNull();
    await act(async () => {
      document.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    expect(container.querySelector('#search-dropdown')).toBeNull();
  });

  it('retains the curated temporal progression in mock mode', async () => {
    vi.mocked(fetchHealth).mockResolvedValue(health('mock'));
    await renderApp();
    await inspectFirstResult();
    await clickButton('View temporal progression');

    expect(container.textContent).toContain('Open image comparison');
    expect(container.textContent).toContain('Progression');
  });
});
