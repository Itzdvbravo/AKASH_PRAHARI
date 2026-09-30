export type Sensor = 'landsat' | 'sentinel-1' | 'sentinel-2' | 'unknown' | 'any';

export interface GeoBBox {
  west: number;
  south: number;
  east: number;
  north: number;
}

export interface Confidence {
  score: number;
  method: string;
  calibrated: boolean;
}

export interface TileRef {
  tile_id: string;
  location_id: string;
  date: string;
  sensor: Sensor;
}

export interface HealthResponse {
  status: string;
  version: string;
  embedding_model: string;
  change_detector: string;
  faiss_index_size: number;
  db_tiles: number;
}

export interface SearchFilters {
  location?: string;
  date_from?: string;
  date_to?: string;
  sensor?: Sensor;
}

export interface SearchRequest {
  query_type: 'text' | 'image';
  query_text?: string;
  query_image_b64?: string;
  filters?: SearchFilters;
  top_k?: number;
}

export interface SearchResultItem {
  rank: number;
  tile_ref: TileRef;
  confidence: Confidence;
  thumbnail_url: string;
  available_dates: string[];
  geo_bbox: GeoBBox;
}

export interface SearchResponse {
  query_id: string;
  results: SearchResultItem[];
  total: number;
  query_ms: number;
}

export interface ComparisonRequest {
  location_id: string;
  tile_id: string;
  date_before: string;
  date_after: string;
  sensor: Sensor;
}

export interface ComparisonResponse {
  comparison_id: string;
  before: {
    tile_ref: TileRef;
    image_url: string;
    cloud_cover_pct: number | null;
  };
  after: {
    tile_ref: TileRef;
    image_url: string;
    cloud_cover_pct: number | null;
  };
  coregistered: boolean;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  confidence: Confidence;
}

export interface AnalystSummary {
  location: string;
  date_before: string;
  date_after: string;
  change_type: string;
  earliest_detectable_change: string | null;
  confidence: Confidence;
  changed_pixel_fraction: number;
  source_provenance: string;
  detector: string;
}

export interface ChangeDetectionRequest {
  comparison_id?: string;
  location_id?: string;
  tile_id?: string;
  date_before?: string;
  date_after?: string;
}

export interface ChangeDetectionResponse {
  job_id: string;
  status: 'completed' | 'processing' | 'failed';
  mask_url: string;
  bounding_boxes: BoundingBox[];
  summary: AnalystSummary;
  processing_ms: number;
}

export type TemporalIconType = 'leaf' | 'crane' | 'building' | 'detection';
export type TemporalTagVariant = 'green' | 'yellow' | 'neutral' | 'red';

export interface TemporalStageItem {
  id: string;
  year: string;
  date: string;
  imageUrl: string;
  title: string;
  subtitle?: string;
  iconType: TemporalIconType;
  variant: TemporalTagVariant;
  sensor?: string;
  cloudCoverPct?: number;
}

export interface TemporalPolygon {
  points: string; // e.g. "49,32 62.5,37 56,45 49.5,39.5" in viewBox 0 0 100 100
  label?: string;
}

export interface TemporalDetectedChanges {
  title: string;
  subRange: string;
  imageUrl: string;
  chipTitle: string;
  chipSubtitle?: string;
  iconType: TemporalIconType;
  variant: TemporalTagVariant;
  polygons: TemporalPolygon[];
  boundingBoxes?: BoundingBox[];
}

export interface TemporalProgressionData {
  locationId: string;
  locationLabel: string;
  country: string;
  timeRange: string;
  changeType: string;
  earliestSupportedChange: string;
  confidence: number;
  stages: TemporalStageItem[];
  detectedChanges: TemporalDetectedChanges;
  description: string;
  changedPixelFraction: number;
}

