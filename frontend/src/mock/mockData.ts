import {
  HealthResponse,
  SearchResultItem,
  ComparisonResponse,
  ChangeDetectionResponse
} from '../types/api.types';

// Helper to generate clean, high-contrast SVG data URLs for mock satellite tiles
function createSvgDataUrl(title: string, date: string, type: 'before' | 'after' | 'mask' | 'berlin'): string {
  if (type === 'mask') {
    // Binary change mask (black with crisp neon/orange change region)
    const svg = `
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="400" height="400">
        <rect width="400" height="400" fill="transparent"/>
        <rect x="230" y="220" width="110" height="100" rx="4" fill="#f43f5e" fill-opacity="0.85" stroke="#ffe4e6" stroke-width="2"/>
        <text x="240" y="245" fill="#ffffff" font-family="monospace" font-size="12" font-weight="bold">CHANGE: +27.5ha</text>
        <text x="240" y="265" fill="#ffe4e6" font-family="monospace" font-size="10">CONF: 94.2%</text>
      </svg>
    `.trim();
    return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
  }

  if (type === 'berlin') {
    const svg = `
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="400" height="400">
        <defs>
          <linearGradient id="bgBerlin" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#1e293b"/>
            <stop offset="100%" stop-color="#0f172a"/>
          </linearGradient>
          <pattern id="gridBerlin" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#334155" stroke-width="0.7"/>
          </pattern>
        </defs>
        <rect width="400" height="400" fill="url(#bgBerlin)"/>
        <rect width="400" height="400" fill="url(#gridBerlin)"/>
        <!-- River Spree -->
        <path d="M 0,260 C 120,240 220,290 400,270" fill="none" stroke="#0284c7" stroke-width="24" stroke-linecap="round"/>
        <!-- Forest Tiergarten -->
        <rect x="40" y="50" width="140" height="120" rx="10" fill="#14532d" opacity="0.85"/>
        <circle cx="110" cy="110" r="28" fill="#166534"/>
        <!-- Urban infrastructure blocks -->
        <rect x="220" y="60" width="130" height="90" fill="#475569" stroke="#64748b" stroke-width="1.5"/>
        <text x="20" y="30" fill="#94a3b8" font-family="sans-serif" font-size="13" font-weight="600">${title}</text>
        <text x="20" y="380" fill="#64748b" font-family="monospace" font-size="11">DATE: ${date} | SENTINEL-2</text>
      </svg>
    `.trim();
    return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
  }

  const isAfter = type === 'after';
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="400" height="400">
      <defs>
        <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#14281d"/>
          <stop offset="100%" stop-color="#091811"/>
        </linearGradient>
        <pattern id="grid" width="30" height="30" patternUnits="userSpaceOnUse">
          <path d="M 30 0 L 0 0 0 30" fill="none" stroke="#1e3a29" stroke-width="0.6"/>
        </pattern>
      </defs>
      <!-- Base Field -->
      <rect width="400" height="400" fill="url(#bgGrad)"/>
      <rect width="400" height="400" fill="url(#grid)"/>

      <!-- River Seine Meander -->
      <path d="M 0,160 Q 180,120 220,200 T 400,180" fill="none" stroke="#0369a1" stroke-width="32" stroke-linecap="round"/>

      <!-- Agricultural / Forest patches -->
      <rect x="30" y="30" width="110" height="90" rx="8" fill="#1b4332" opacity="0.9"/>
      <rect x="270" y="40" width="90" height="90" rx="6" fill="#2d6a4f" opacity="0.7"/>

      <!-- Original Settlement -->
      <rect x="50" y="240" width="100" height="90" rx="4" fill="#334155" stroke="#475569" stroke-width="1"/>
      <line x1="60" y1="285" x2="140" y2="285" stroke="#64748b" stroke-width="2"/>
      <line x1="100" y1="245" x2="100" y2="325" stroke="#64748b" stroke-width="2"/>

      ${
        isAfter
          ? `
        <!-- NEW EXPANSION IN AFTER IMAGE -->
        <rect x="230" y="220" width="110" height="100" rx="4" fill="#dc2626" fill-opacity="0.8" stroke="#fca5a5" stroke-width="2"/>
        <line x1="240" y1="270" x2="330" y2="270" stroke="#fef2f2" stroke-width="2"/>
        <line x1="285" y1="230" x2="285" y2="310" stroke="#fef2f2" stroke-width="2"/>
        <circle cx="285" cy="270" r="16" fill="#f87171" opacity="0.6"/>
        <text x="240" y="340" fill="#fca5a5" font-family="monospace" font-size="11" font-weight="bold">NEW ZONE DETECTED</text>
      `
          : `
        <!-- Unaltered Agricultural field in BEFORE image -->
        <rect x="230" y="220" width="110" height="100" rx="4" fill="#1b4332" stroke="#2d6a4f" stroke-width="1"/>
        <circle cx="285" cy="270" r="24" fill="#2d6a4f" opacity="0.5"/>
      `
      }

      <text x="20" y="30" fill="#e2e8f0" font-family="sans-serif" font-size="13" font-weight="600">${title}</text>
      <text x="20" y="380" fill="#94a3b8" font-family="monospace" font-size="11">DATE: ${date} | SENTINEL-2</text>
    </svg>
  `.trim();
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

export const MOCK_HEALTH: HealthResponse = {
  status: 'healthy',
  version: '0.1.0-alpha',
  embedding_model: 'Clay-v1-GeoSpatial (ViT-B/16)',
  change_detector: 'Siamese-BiDiff-ResNet50',
  faiss_index_size: 4820,
  db_tiles: 12450
};

export const MOCK_SEARCH_RESULTS: SearchResultItem[] = [
  {
    rank: 1,
    tile_ref: {
      tile_id: 'paris_tile_001_s2',
      location_id: 'paris',
      date: '2020-06-15',
      sensor: 'sentinel-2'
    },
    confidence: {
      score: 0.942,
      method: 'cosine_faiss',
      calibrated: true
    },
    thumbnail_url: createSvgDataUrl('Paris Sector 01 (Post-Expansion)', '2020-06-15', 'after'),
    available_dates: ['2018-03-10', '2019-04-12', '2020-06-15', '2021-09-02'],
    geo_bbox: {
      west: 2.29,
      south: 48.85,
      east: 2.31,
      north: 48.87
    }
  },
  {
    rank: 2,
    tile_ref: {
      tile_id: 'berlin_tile_004_s2',
      location_id: 'berlin',
      date: '2019-05-20',
      sensor: 'sentinel-2'
    },
    confidence: {
      score: 0.887,
      method: 'cosine_faiss',
      calibrated: true
    },
    thumbnail_url: createSvgDataUrl('Berlin Mitte Infrastructure', '2019-05-20', 'berlin'),
    available_dates: ['2018-06-10', '2019-05-20', '2020-08-11'],
    geo_bbox: {
      west: 13.38,
      south: 52.51,
      east: 13.41,
      north: 52.53
    }
  },
  {
    rank: 3,
    tile_ref: {
      tile_id: 'paris_tile_001_pre',
      location_id: 'paris',
      date: '2018-03-10',
      sensor: 'sentinel-2'
    },
    confidence: {
      score: 0.854,
      method: 'cosine_faiss',
      calibrated: true
    },
    thumbnail_url: createSvgDataUrl('Paris Sector 01 (Baseline)', '2018-03-10', 'before'),
    available_dates: ['2018-03-10', '2019-04-12', '2020-06-15'],
    geo_bbox: {
      west: 2.29,
      south: 48.85,
      east: 2.31,
      north: 48.87
    }
  }
];

export const MOCK_COMPARISON_PARIS: ComparisonResponse = {
  comparison_id: 'cmp_paris_2018_2020',
  before: {
    tile_ref: {
      tile_id: 'paris_2018_03_10',
      location_id: 'paris',
      date: '2018-03-10',
      sensor: 'sentinel-2'
    },
    image_url: createSvgDataUrl('Paris Baseline Observation', '2018-03-10', 'before'),
    cloud_cover_pct: 1.2
  },
  after: {
    tile_ref: {
      tile_id: 'paris_2020_06_15',
      location_id: 'paris',
      date: '2020-06-15',
      sensor: 'sentinel-2'
    },
    image_url: createSvgDataUrl('Paris Post-Expansion Observation', '2020-06-15', 'after'),
    cloud_cover_pct: 0.4
  },
  coregistered: true
};

export const MOCK_CHANGE_DETECTION_PARIS: ChangeDetectionResponse = {
  job_id: 'job_cd_paris_9921',
  status: 'completed',
  mask_url: createSvgDataUrl('Paris Change Mask', 'Delta', 'mask'),
  bounding_boxes: [
    {
      x: 230,
      y: 220,
      width: 110,
      height: 100,
      label: 'New Urban Construction / Impervious Surface',
      confidence: {
        score: 0.942,
        method: 'siamese_diff_v2',
        calibrated: true
      }
    }
  ],
  summary: {
    location: 'Paris, France (Sector 01)',
    date_before: '2018-03-10',
    date_after: '2020-06-15',
    change_type: 'Urban Expansion / New Construction',
    earliest_detectable_change: '2019-04-12',
    confidence: {
      score: 0.942,
      method: 'calibrated_ensemble',
      calibrated: true
    },
    changed_pixel_fraction: 0.0688,
    source_provenance: 'Sentinel-2 L2A BOA Reflectance',
    detector: 'Siamese-BiDiff-ResNet50 v1.4'
  },
  processing_ms: 184
};
