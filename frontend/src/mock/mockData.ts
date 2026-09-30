import {
  HealthResponse,
  SearchResultItem,
  ComparisonResponse,
  ChangeDetectionResponse,
  TemporalProgressionData,
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
    id: 'kaziranga',
    label: 'Kaziranga NP',
    country: 'Assam',
    coords: { lat: 26.5775, lon: 93.1711 },
    date1: '2022',
    date2: '2025',
    beforeUrl: '/satellite/kaziranga/2022.jpg',
    afterUrl: '/satellite/kaziranga/2025.jpg',
    changeType: 'Vegetation → Buildings',
    changeDesc: 'Dense natural forest vegetation replaced by institutional facilities and residential building complexes. Initial clearing and foundation construction began in early 2024.',
    changedFraction: 0.091,
    confidence: 0.91,
    bbox: { west: 93.15, south: 26.56, east: 93.19, north: 26.59 }
  },
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
    date1: '2018-03-10',
    date2: '2020-06-15',
    beforeUrl: createSvgDataUrl('Paris Sector 01 (Baseline)', '2018-03-10', 'before'),
    afterUrl: createSvgDataUrl('Paris Sector 01 (Post-Expansion)', '2020-06-15', 'after'),
    changeType: 'Urban Expansion / New Construction',
    changeDesc: 'New urban construction and impervious surface expansion detected in Sector 01.',
    changedFraction: 0.0688,
    confidence: 0.942,
    bbox: { west: 2.29, south: 48.85, east: 2.31, north: 48.87 }
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
    id: 'q_kaziranga_buildings',
    query: 'Show vegetation to building development in Kaziranga NP Assam 2022 \u2013 2025',
    locationIds: ['kaziranga'],
    category: 'construction',
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
  const findPreset = (id: string) => QUERY_PRESETS.find(p => p.id === id) ?? QUERY_PRESETS[0];
  if (q.includes('kaziranga') || q.includes('assam')) return findPreset('q_kaziranga_buildings');
  if (q.includes('dubai') || q.includes('uae')) return findPreset('q_dubai_buildings');
  if (q.includes('asia') || q.includes('china') || q.includes('hong kong') || q.includes('chongqing')) return findPreset('q_asia_urban');
  if (q.includes('norcia') || q.includes('earthquake') || q.includes('disaster') || q.includes('beirut')) return findPreset('q_norcia_disaster');
  if (q.includes('las vegas') || q.includes('cupertino') || q.includes('suburban') || q.includes('silicon')) return findPreset('q_las_vegas_suburban');
  if (q.includes('brazil') || q.includes('brasilia') || q.includes('deforest')) return findPreset('q_brazil_deforestation');
  if (q.includes('coastal') || q.includes('reclamation') || q.includes('marine')) return findPreset('q_coastal');
  if (q.includes('paris') || q.includes('europe')) return findPreset('q_paris_urban');
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

// ─── Temporal Progression Data ───────────────────────────────────────────────

export function getTemporalProgression(locationId: string): TemporalProgressionData {
  if (locationId === 'kaziranga') {
    return {
      locationId: 'kaziranga',
      locationLabel: 'Kaziranga NP, Assam',
      country: 'India',
      timeRange: '2022 \u2192 2025',
      changeType: 'Vegetation \u2192 Buildings',
      earliestSupportedChange: '2024',
      confidence: 0.91,
      stages: [
        {
          id: 'kaziranga-2022',
          year: '2022',
          date: '2022-03-24',
          imageUrl: '/satellite/kaziranga/2022.jpg',
          title: 'Vegetation (no buildings)',
          iconType: 'leaf',
          variant: 'green',
          sensor: 'Sentinel-2 L2A',
          cloudCoverPct: 0.8,
        },
        {
          id: 'kaziranga-2024',
          year: '2024',
          date: '2024-04-12',
          imageUrl: '/satellite/kaziranga/2024.jpg',
          title: 'Buildings appearing',
          subtitle: '(initial construction)',
          iconType: 'crane',
          variant: 'yellow',
          sensor: 'Sentinel-2 L2A',
          cloudCoverPct: 1.2,
        },
        {
          id: 'kaziranga-2025',
          year: '2025',
          date: '2025-02-18',
          imageUrl: '/satellite/kaziranga/2025.jpg',
          title: 'Buildings',
          subtitle: '(construction completed)',
          iconType: 'building',
          variant: 'neutral',
          sensor: 'Sentinel-2 L2A',
          cloudCoverPct: 0.4,
        },
      ],
      detectedChanges: {
        title: 'Detected Changes',
        subRange: '(2022 vs 2025)',
        imageUrl: '/satellite/kaziranga/2025.jpg',
        chipTitle: 'Changed areas',
        chipSubtitle: '(new buildings)',
        iconType: 'detection',
        variant: 'red',
        polygons: [
          { points: '48,31 63,36 57,45 48,39', label: 'Institutional Wing' },
          { points: '57,46 72,44 74,50 64,56 57,51', label: 'Main Complex' },
          { points: '39,38 49,45 45,50 38,44', label: 'Residential Wing 1' },
          { points: '43,56 57,64 52,70 43,63', label: 'Residential Wing 2' },
          { points: '26,42 36,42 36,52 26,52', label: 'Facilities Hub' },
          { points: '30,51 41,56 37,63 29,58', label: 'Support Block' },
        ],
        boundingBoxes: [
          {
            x: 95,
            y: 110,
            width: 210,
            height: 190,
            label: 'Primary Development Zone',
            confidence: { score: 0.91, method: 'calibrated_ensemble', calibrated: true },
          },
        ],
      },
      description:
        'Dense natural forest vegetation replaced by institutional facilities and residential building complexes. Initial clearing and foundation construction began in early 2024.',
      changedPixelFraction: 0.091,
    };
  }

  const loc = LOCATIONS.find(l => l.id === locationId) || LOCATIONS[0];
  const year1 = loc.date1.slice(0, 4);
  const year2 = loc.date2.slice(0, 4);
  const midYear = String(Math.round((parseInt(year1) + parseInt(year2)) / 2));

  return {
    locationId: loc.id,
    locationLabel: `${loc.label}, ${loc.country}`,
    country: loc.country,
    timeRange: `${year1} \u2192 ${year2}`,
    changeType: loc.changeType,
    earliestSupportedChange: midYear,
    confidence: loc.confidence,
    stages: [
      {
        id: `${loc.id}-${year1}`,
        year: year1,
        date: loc.date1,
        imageUrl: loc.beforeUrl,
        title: 'Baseline State',
        subtitle: '(pre-event reference)',
        iconType: 'leaf',
        variant: 'green',
        sensor: 'Sentinel-2 L2A',
        cloudCoverPct: 0.5,
      },
      {
        id: `${loc.id}-${midYear}`,
        year: midYear,
        date: `${midYear}-06-15`,
        imageUrl: loc.beforeUrl,
        title: 'Initial Transitions',
        subtitle: '(detected onset)',
        iconType: 'crane',
        variant: 'yellow',
        sensor: 'Sentinel-2 L2A',
        cloudCoverPct: 1.0,
      },
      {
        id: `${loc.id}-${year2}`,
        year: year2,
        date: loc.date2,
        imageUrl: loc.afterUrl,
        title: 'Target State',
        subtitle: '(current observation)',
        iconType: 'building',
        variant: 'neutral',
        sensor: 'Sentinel-2 L2A',
        cloudCoverPct: 0.3,
      },
    ],
    detectedChanges: {
      title: 'Detected Changes',
      subRange: `(${year1} vs ${year2})`,
      imageUrl: loc.afterUrl,
      chipTitle: 'Changed areas',
      chipSubtitle: `(${loc.changeType.split('/')[0].trim().toLowerCase()})`,
      iconType: 'detection',
      variant: 'red',
      polygons: [
        { points: '25,20 50,18 55,42 22,40', label: 'Primary Sector' },
        { points: '52,50 82,48 85,75 55,78', label: 'Secondary Sector' },
        { points: '30,62 48,60 46,78 28,76', label: 'Expansion Wing' },
      ],
      boundingBoxes: [
        {
          x: 60,
          y: 55,
          width: 140,
          height: 130,
          label: loc.changeType,
          confidence: { score: loc.confidence, method: 'calibrated_ensemble', calibrated: true },
        },
      ],
    },
    description: loc.changeDesc,
    changedPixelFraction: loc.changedFraction,
  };
}

// ─── Default / Initial State ──────────────────────────────────────────────────

export const DEFAULT_LOCATION = LOCATIONS.find(l => l.id === 'kaziranga') || LOCATIONS[0];

export const MOCK_COMPARISON_PARIS: ComparisonResponse = buildComparison(DEFAULT_LOCATION);
export const MOCK_CHANGE_DETECTION_PARIS: ChangeDetectionResponse = buildChangeDetection(DEFAULT_LOCATION);

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
      tile_id: 'kaziranga_tile_001_s2',
      location_id: 'kaziranga',
      date: '2025-02-18',
      sensor: 'sentinel-2'
    },
    confidence: {
      score: 0.958,
      method: 'cosine_faiss',
      calibrated: true
    },
    thumbnail_url: '/satellite/kaziranga/2025.jpg',
    available_dates: ['2022-03-24', '2024-04-12', '2025-02-18'],
    geo_bbox: {
      west: 93.15,
      south: 26.56,
      east: 93.19,
      north: 26.59
    }
  },
  {
    rank: 2,
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
    rank: 3,
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
    rank: 4,
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



