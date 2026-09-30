import {
  HealthResponse,
  SearchResultItem,
  ComparisonResponse,
  ChangeDetectionResponse
} from '../types/api.types';

// ─── Location Metadata ───────────────────────────────────────────────────────

export interface LocationMeta {
  id: string;
  label: string;
  country: string;
  coords: { lat: number; lon: number };
  date1: string;   // YYYY-MM-DD  (epoch 1)
  date2: string;   // YYYY-MM-DD  (epoch 2)
  beforeUrl: string;
  afterUrl: string;
  changeType: string;
  changeDesc: string;
  changedFraction: number;   // 0-1
  confidence: number;        // 0-1
  bbox: { west: number; south: number; east: number; north: number };
}

export const LOCATIONS: LocationMeta[] = [
  {
    id: 'dubai',
    label: 'Dubai',
    country: 'UAE',
    coords: { lat: 25.2048, lon: 55.2708 },
    date1: '2015-12-11',
    date2: '2018-03-30',
    beforeUrl: '/satellite/dubai/before.png',
    afterUrl: '/satellite/dubai/after.png',
    changeType: 'Urban Expansion / New Construction',
    changeDesc: 'Rapid high-rise construction and land reclamation detected in coastal development zones.',
    changedFraction: 0.1142,
    confidence: 0.961,
    bbox: { west: 55.26, south: 25.19, east: 55.28, north: 25.22 }
  },
  {
    id: 'abudhabi',
    label: 'Abu Dhabi',
    country: 'UAE',
    coords: { lat: 24.4539, lon: 54.3773 },
    date1: '2016-01-20',
    date2: '2018-03-28',
    beforeUrl: '/satellite/abudhabi/before.png',
    afterUrl: '/satellite/abudhabi/after.png',
    changeType: 'Infrastructure Development',
    changeDesc: 'New road networks and residential complexes expanded into previously undeveloped desert areas.',
    changedFraction: 0.0891,
    confidence: 0.934,
    bbox: { west: 54.36, south: 24.44, east: 54.40, north: 24.47 }
  },
  {
    id: 'mumbai',
    label: 'Mumbai',
    country: 'India',
    coords: { lat: 19.0760, lon: 72.8777 },
    date1: '2015-11-30',
    date2: '2018-03-19',
    beforeUrl: '/satellite/mumbai/before.png',
    afterUrl: '/satellite/mumbai/after.png',
    changeType: 'Urban Densification',
    changeDesc: 'Slum redevelopment and new high-density residential towers emerged in suburban fringes.',
    changedFraction: 0.0763,
    confidence: 0.912,
    bbox: { west: 72.86, south: 19.06, east: 72.90, north: 19.09 }
  },
  {
    id: 'lasvegas',
    label: 'Las Vegas',
    country: 'USA',
    coords: { lat: 36.1699, lon: -115.1398 },
    date1: '2015-08-20',
    date2: '2018-02-05',
    beforeUrl: '/satellite/lasvegas/before.png',
    afterUrl: '/satellite/lasvegas/after.png',
    changeType: 'Suburban Growth',
    changeDesc: 'New suburban residential tracts and commercial strips appeared at the urban boundary.',
    changedFraction: 0.0624,
    confidence: 0.887,
    bbox: { west: -115.15, south: 36.16, east: -115.12, north: 36.18 }
  },
  {
    id: 'chongqing',
    label: 'Chongqing',
    country: 'China',
    coords: { lat: 29.5630, lon: 106.5516 },
    date1: '2017-04-14',
    date2: '2018-04-02',
    beforeUrl: '/satellite/chongqing/before.png',
    afterUrl: '/satellite/chongqing/after.png',
    changeType: 'Mega-Urban Expansion',
    changeDesc: 'Massive new urban districts with towers and transit infrastructure built over farmland.',
    changedFraction: 0.1356,
    confidence: 0.978,
    bbox: { west: 106.54, south: 29.55, east: 106.57, north: 29.58 }
  },
  {
    id: 'beirut',
    label: 'Beirut',
    country: 'Lebanon',
    coords: { lat: 33.8938, lon: 35.5018 },
    date1: '2015-08-20',
    date2: '2017-10-03',
    beforeUrl: '/satellite/beirut/before.png',
    afterUrl: '/satellite/beirut/after.png',
    changeType: 'Post-Conflict Reconstruction',
    changeDesc: 'Reconstruction activity and new building footprints detected across the urban core.',
    changedFraction: 0.0482,
    confidence: 0.856,
    bbox: { west: 35.49, south: 33.88, east: 35.52, north: 33.91 }
  },
  {
    id: 'paris',
    label: 'Paris',
    country: 'France',
    coords: { lat: 48.8566, lon: 2.3522 },
    date1: '2016-11-30',
    date2: '2017-11-07',
    beforeUrl: '/satellite/paris/before.png',
    afterUrl: '/satellite/paris/after.png',
    changeType: 'Urban Renewal',
    changeDesc: 'Renovation projects and new mixed-use buildings detected in peri-urban zones.',
    changedFraction: 0.0312,
    confidence: 0.823,
    bbox: { west: 2.34, south: 48.85, east: 2.37, north: 48.87 }
  },
  {
    id: 'cupertino',
    label: 'Cupertino',
    country: 'USA',
    coords: { lat: 37.3230, lon: -122.0322 },
    date1: '2015-09-18',
    date2: '2018-03-26',
    beforeUrl: '/satellite/cupertino/before.png',
    afterUrl: '/satellite/cupertino/after.png',
    changeType: 'Tech Campus Construction',
    changeDesc: 'Large corporate campus buildings and parking structures appeared in Silicon Valley suburban area.',
    changedFraction: 0.0521,
    confidence: 0.944,
    bbox: { west: -122.04, south: 37.31, east: -122.01, north: 37.34 }
  },
  {
    id: 'hongkong',
    label: 'Hong Kong',
    country: 'China',
    coords: { lat: 22.3193, lon: 114.1694 },
    date1: '2016-09-27',
    date2: '2018-03-23',
    beforeUrl: '/satellite/hongkong/before.png',
    afterUrl: '/satellite/hongkong/after.png',
    changeType: 'Coastal Reclamation',
    changeDesc: 'Land reclamation and new high-density buildings detected along the harbour front.',
    changedFraction: 0.0688,
    confidence: 0.921,
    bbox: { west: 114.15, south: 22.30, east: 114.19, north: 22.34 }
  },
  {
    id: 'brasilia',
    label: 'Brasilia',
    country: 'Brazil',
    coords: { lat: -15.7942, lon: -47.8822 },
    date1: '2015-09-16',
    date2: '2017-10-17',
    beforeUrl: '/satellite/brasilia/before.png',
    afterUrl: '/satellite/brasilia/after.png',
    changeType: 'Deforestation & Urban Spread',
    changeDesc: 'Forest clearance and new satellite town development detected at the urban periphery.',
    changedFraction: 0.0934,
    confidence: 0.898,
    bbox: { west: -47.90, south: -15.81, east: -47.86, north: -15.78 }
  },
  {
    id: 'beihai',
    label: 'Beihai',
    country: 'China',
    coords: { lat: 21.4733, lon: 109.1200 },
    date1: '2016-12-09',
    date2: '2018-03-09',
    beforeUrl: '/satellite/beihai/before.png',
    afterUrl: '/satellite/beihai/after.png',
    changeType: 'Coastal Development',
    changeDesc: 'Rapid coastal resort and industrial zone expansion into shallow marine environments.',
    changedFraction: 0.1021,
    confidence: 0.947,
    bbox: { west: 109.10, south: 21.46, east: 109.14, north: 21.49 }
  },
  {
    id: 'norcia',
    label: 'Norcia',
    country: 'Italy',
    coords: { lat: 42.7942, lon: 13.0955 },
    date1: '2015-07-11',
    date2: '2017-10-18',
    beforeUrl: '/satellite/norcia/before.png',
    afterUrl: '/satellite/norcia/after.png',
    changeType: 'Earthquake Damage',
    changeDesc: 'Structural collapse and debris fields detected following the 2016 Central Italy earthquake.',
    changedFraction: 0.0347,
    confidence: 0.976,
    bbox: { west: 13.08, south: 42.78, east: 13.11, north: 42.81 }
  },
];

// ─── Hardcoded NLP Query Presets ─────────────────────────────────────────────

export interface QueryPreset {
  id: string;
  query: string;
  locationIds: string[];
  category: 'construction' | 'disaster' | 'deforestation' | 'infrastructure' | 'all';
}

export const QUERY_PRESETS: QueryPreset[] = [
  {
    id: 'q_all',
    query: '',
    locationIds: LOCATIONS.map(l => l.id),
    category: 'all',
  },
  {
    id: 'q_dubai_buildings',
    query: 'Show me where new buildings appeared between 2015 \u2013 2018 in Dubai',
    locationIds: ['dubai', 'abudhabi'],
    category: 'construction',
  },
  {
    id: 'q_asia_urban',
    query: 'Detect rapid urban expansion in Asian megacities 2016 \u2013 2018',
    locationIds: ['chongqing', 'beihai', 'hongkong', 'mumbai'],
    category: 'construction',
  },
  {
    id: 'q_norcia_disaster',
    query: 'Show earthquake damage in Norcia Italy after 2016',
    locationIds: ['norcia', 'beirut'],
    category: 'disaster',
  },
  {
    id: 'q_las_vegas_suburban',
    query: 'Find suburban sprawl and new residential zones in the American West',
    locationIds: ['lasvegas', 'cupertino'],
    category: 'infrastructure',
  },
  {
    id: 'q_brazil_deforestation',
    query: 'Identify deforestation and land clearing near Brasilia 2015 \u2013 2017',
    locationIds: ['brasilia'],
    category: 'deforestation',
  },
  {
    id: 'q_coastal',
    query: 'Show coastal land reclamation and marine zone development',
    locationIds: ['dubai', 'hongkong', 'beihai', 'abudhabi'],
    category: 'construction',
  },
  {
    id: 'q_paris_urban',
    query: 'Urban renewal and new construction in European cities 2016 \u2013 2018',
    locationIds: ['paris', 'beirut'],
    category: 'infrastructure',
  },
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function locToSearchItem(loc: LocationMeta, rank: number): SearchResultItem {
  return {
    rank,
    tile_ref: {
      tile_id: `${loc.id}_tile_s2`,
      location_id: loc.id,
      date: loc.date2,
      sensor: 'sentinel-2',
    },
    confidence: {
      score: loc.confidence,
      method: 'cosine_faiss',
      calibrated: true,
    },
    thumbnail_url: loc.beforeUrl,
    available_dates: [loc.date1, loc.date2],
    geo_bbox: loc.bbox,
  };
}

export function getSearchResults(queryId: string): SearchResultItem[] {
  const preset = QUERY_PRESETS.find(p => p.id === queryId) ?? QUERY_PRESETS[0];
  return preset.locationIds
    .map(id => LOCATIONS.find(l => l.id === id)!)
    .filter(Boolean)
    .map((loc, i) => locToSearchItem(loc, i + 1));
}

export function matchQueryToPreset(query: string): QueryPreset {
  if (!query.trim()) return QUERY_PRESETS[0];
  const q = query.toLowerCase();
  for (const preset of QUERY_PRESETS) {
    if (preset.query && preset.query.toLowerCase() === q) return preset;
  }
  if (q.includes('dubai') || q.includes('uae')) return QUERY_PRESETS[1];
  if (q.includes('asia') || q.includes('china') || q.includes('hong kong') || q.includes('chongqing')) return QUERY_PRESETS[2];
  if (q.includes('norcia') || q.includes('earthquake') || q.includes('disaster') || q.includes('beirut')) return QUERY_PRESETS[3];
  if (q.includes('las vegas') || q.includes('cupertino') || q.includes('suburban') || q.includes('silicon')) return QUERY_PRESETS[4];
  if (q.includes('brazil') || q.includes('brasilia') || q.includes('deforest')) return QUERY_PRESETS[5];
  if (q.includes('coastal') || q.includes('reclamation') || q.includes('marine')) return QUERY_PRESETS[6];
  if (q.includes('paris') || q.includes('europe')) return QUERY_PRESETS[7];
  return QUERY_PRESETS[0];
}

export function buildComparison(loc: LocationMeta): ComparisonResponse {
  return {
    comparison_id: `cmp_${loc.id}_${loc.date1}_${loc.date2}`,
    before: {
      tile_ref: {
        tile_id: `${loc.id}_${loc.date1.replace(/-/g, '')}_s2`,
        location_id: loc.id,
        date: loc.date1,
        sensor: 'sentinel-2',
      },
      image_url: loc.beforeUrl,
      cloud_cover_pct: parseFloat((Math.random() * 3).toFixed(1)),
    },
    after: {
      tile_ref: {
        tile_id: `${loc.id}_${loc.date2.replace(/-/g, '')}_s2`,
        location_id: loc.id,
        date: loc.date2,
        sensor: 'sentinel-2',
      },
      image_url: loc.afterUrl,
      cloud_cover_pct: parseFloat((Math.random() * 2).toFixed(1)),
    },
    coregistered: true,
  };
}

export function buildChangeDetection(loc: LocationMeta): ChangeDetectionResponse {
  return {
    job_id: `job_cd_${loc.id}_${Date.now()}`,
    status: 'completed',
    mask_url: loc.afterUrl,
    bounding_boxes: [
      {
        x: 60,
        y: 55,
        width: 130,
        height: 120,
        label: loc.changeType,
        confidence: { score: loc.confidence, method: 'siamese_diff_v2', calibrated: true },
      },
      {
        x: 230,
        y: 200,
        width: 90,
        height: 85,
        label: 'Secondary Zone',
        confidence: { score: loc.confidence - 0.08, method: 'siamese_diff_v2', calibrated: true },
      },
    ],
    summary: {
      location: `${loc.label}, ${loc.country}`,
      date_before: loc.date1,
      date_after: loc.date2,
      change_type: loc.changeType,
      earliest_detectable_change: loc.date1,
      confidence: { score: loc.confidence, method: 'calibrated_ensemble', calibrated: true },
      changed_pixel_fraction: loc.changedFraction,
      source_provenance: 'Sentinel-2 L2A BOA Reflectance (OSCD Dataset)',
      detector: 'Siamese-BiDiff-ResNet50 v1.4',
    },
    processing_ms: Math.floor(Math.random() * 300 + 100),
  };
}

// ─── Default / Initial State ──────────────────────────────────────────────────

export const DEFAULT_LOCATION = LOCATIONS.find(l => l.id === 'dubai')!;

export const MOCK_COMPARISON_PARIS: ComparisonResponse = buildComparison(DEFAULT_LOCATION);
export const MOCK_CHANGE_DETECTION_PARIS: ChangeDetectionResponse = buildChangeDetection(DEFAULT_LOCATION);

export const MOCK_HEALTH: HealthResponse = {
  status: 'healthy',
  version: '0.1.0-alpha',
  embedding_model: 'Clay-v1-GeoSpatial (ViT-B/16)',
  change_detector: 'Siamese-BiDiff-ResNet50',
  faiss_index_size: 4820,
  db_tiles: 12450,
};
