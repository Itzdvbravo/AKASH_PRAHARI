import React, { useState, useEffect, useRef } from 'react';
import {
  TemporalProgressionData,
  TemporalStageItem,
  TemporalIconType,
  TemporalTagVariant,
} from '../../types/api.types';
import { LOCATIONS, getTemporalProgression } from '../../mock/mockData';

interface TemporalProgressionViewerProps {
  locationId?: string;
  onLocationChange?: (locationId: string) => void;
  isLoading?: boolean;
}

export const TemporalProgressionViewer: React.FC<TemporalProgressionViewerProps> = ({
  locationId = 'dubai',
  onLocationChange,
  isLoading = false,
}) => {
  const [selectedLocId, setSelectedLocId] = useState<string>(locationId);
  const [viewMode, setViewMode] = useState<'progression' | 'slider' | 'sideBySide'>('progression');
  const [overlayStyle, setOverlayStyle] = useState<'polygon' | 'bbox' | 'none'>('polygon');
  const maskOpacity = 0.85;
  const [themeMode, setThemeMode] = useState<'lightCard' | 'cyberDark'>('lightCard');

  // Slider state
  const [sliderPos, setSliderPos] = useState<number>(50);

  // Timelapse playback state
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [activeStep, setActiveStep] = useState<number>(0);
  const playTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Modal zoom state
  const [zoomedItem, setZoomedItem] = useState<{
    title: string;
    subtitle?: string;
    imageUrl: string;
    date: string;
    isDetected?: boolean;
  } | null>(null);

  // Synchronize when parent passes locationId
  useEffect(() => {
    setSelectedLocId(locationId);
  }, [locationId]);

  const data: TemporalProgressionData = getTemporalProgression(selectedLocId);
  const [copyStatus, setCopyStatus] = useState('Copy coordinates');

  const footprintAreaKm2 = (() => {
    const radiusKm = 6371.0088;
    const { west, south, east, north } = data.footprint;
    const radians = (degrees: number) => (degrees * Math.PI) / 180;
    return (radiusKm ** 2) * radians(east - west) *
      (Math.sin(radians(north)) - Math.sin(radians(south)));
  })();
  const changedAreaKm2 = footprintAreaKm2 * data.changedPixelFraction;

  const copyCoordinates = async () => {
    const { latitude, longitude } = data.coordinates;
    try {
      await navigator.clipboard.writeText(`${latitude.toFixed(6)}, ${longitude.toFixed(6)}`);
      setCopyStatus('Coordinates copied');
      window.setTimeout(() => setCopyStatus('Copy coordinates'), 1800);
    } catch {
      setCopyStatus('Clipboard unavailable');
      window.setTimeout(() => setCopyStatus('Copy coordinates'), 1800);
    }
  };

  const exportFootprint = () => {
    const { west, south, east, north } = data.footprint;
    const geojson = {
      type: 'FeatureCollection',
      name: `${data.locationId}_temporal_analysis`,
      features: [{
        type: 'Feature',
        properties: {
          location: data.locationLabel,
          date_range: data.timeRange,
          coordinates: data.coordinates,
          crs: data.spatialReference,
          change_type: data.changeType,
          changed_pixel_fraction: data.changedPixelFraction,
          estimated_footprint_area_km2: Number(footprintAreaKm2.toFixed(3)),
          estimated_changed_area_km2: Number(changedAreaKm2.toFixed(3)),
          source: data.source,
          raster_format: data.rasterFormat,
          raster_metadata_status: data.rasterMetadataStatus,
        },
        geometry: {
          type: 'Polygon',
          coordinates: [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        },
      }],
    };
    const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/geo+json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${data.locationId}-temporal-footprint.geojson`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleSelectLocation = (id: string) => {
    setSelectedLocId(id);
    setActiveStep(0);
    setIsPlaying(false);
    if (onLocationChange) {
      onLocationChange(id);
    }
  };

  // Timelapse animation loop
  useEffect(() => {
    if (isPlaying) {
      playTimerRef.current = setInterval(() => {
        setActiveStep(prev => (prev + 1) % 4);
      }, 1400);
    } else if (playTimerRef.current) {
      clearInterval(playTimerRef.current);
    }
    return () => {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    };
  }, [isPlaying]);

  const handleSliderDrag = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    setSliderPos(Math.max(5, Math.min(95, (x / rect.width) * 100)));
  };

  // Icon Renderers
  const renderHeaderIcon = (type: 'location' | 'calendar' | 'change' | 'clock' | 'shield') => {
    switch (type) {
      case 'location':
        return (
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2a7 7 0 0 0-7 7c0 5.25 7 13 7 13s7-7.75 7-13a7 7 0 0 0-7-7z" />
            <circle cx="12" cy="9" r="2.5" />
          </svg>
        );
      case 'calendar':
        return (
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
            <line x1="16" y1="2" x2="16" y2="6" />
            <line x1="8" y1="2" x2="8" y2="6" />
            <line x1="3" y1="10" x2="21" y2="10" />
            <path d="M8 14h.01M12 14h.01M16 14h.01M8 18h.01M12 18h.01M16 18h.01" />
          </svg>
        );
      case 'change':
        return (
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="m8 3 4 8 5-5 5 15H2L8 3z" />
          </svg>
        );
      case 'clock':
        return (
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
          </svg>
        );
      case 'shield':
        return (
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            <path d="m9 12 2 2 4-4" />
          </svg>
        );
    }
  };

  const renderChipIcon = (type: TemporalIconType) => {
    switch (type) {
      case 'leaf':
        return (
          <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
            <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" />
            <path d="M2 21c0-3 1.85-5.36 5.08-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" fill="none" />
          </svg>
        );
      case 'crane':
        return (
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M2 22h20" />
            <path d="M4 22V7l14-4v6" />
            <path d="M4 11h14" />
            <path d="M18 9v6" />
            <path d="M18 15a2 2 0 1 0 0 4 2 2 0 0 0 0-4z" />
            <path d="M9 22v-7l3-2 3 2v7" />
          </svg>
        );
      case 'building':
        return (
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <rect x="4" y="2" width="16" height="20" rx="2" ry="2" />
            <path d="M9 22v-4h6v4" />
            <path d="M8 6h.01M16 6h.01M12 6h.01M8 10h.01M12 10h.01M16 10h.01M8 14h.01M12 14h.01M16 14h.01" />
          </svg>
        );
      case 'detection':
        return (
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray="3 3">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
          </svg>
        );
    }
  };

  const getChipClass = (variant: TemporalTagVariant) => {
    switch (variant) {
      case 'green':
        return 'temporal-chip-green';
      case 'yellow':
        return 'temporal-chip-yellow';
      case 'neutral':
        return 'temporal-chip-neutral';
      case 'red':
        return 'temporal-chip-red';
    }
  };

  const baselineStage = data.stages[0];
  const targetStage = data.stages[data.stages.length - 1];

  return (
    <div className={`temporal-viewer-wrapper ${themeMode === 'lightCard' ? 'theme-light-card' : 'theme-cyber-dark'}`}>
      {/* ── Control Bar ── */}
      <div className="temporal-control-bar">
        <div className="location-selector-group">
          <span className="control-label">Location:</span>
          <select
            id="temporal-location-dropdown"
            className="temporal-select"
            value={selectedLocId}
            onChange={e => handleSelectLocation(e.target.value)}
          >
            {LOCATIONS.map(loc => (
              <option key={loc.id} value={loc.id}>
                {loc.label}, {loc.country}
              </option>
            ))}
          </select>
        </div>

        <div className="view-mode-tabs">
          <button
            type="button"
            id="tab-progression"
            className={`view-tab-btn ${viewMode === 'progression' ? 'active' : ''}`}
            onClick={() => setViewMode('progression')}
          >
            <span>Timeline Progression</span>
          </button>
          <button
            type="button"
            id="tab-slider"
            className={`view-tab-btn ${viewMode === 'slider' ? 'active' : ''}`}
            onClick={() => setViewMode('slider')}
          >
            <span>Swipe Slider</span>
          </button>
          <button
            type="button"
            id="tab-side-by-side"
            className={`view-tab-btn ${viewMode === 'sideBySide' ? 'active' : ''}`}
            onClick={() => setViewMode('sideBySide')}
          >
            <span>Side-by-Side</span>
          </button>
        </div>

        <div className="toolbar-actions">
          {isLoading && (
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <span className="spinner" /> Updating…
            </span>
          )}
          {viewMode === 'progression' && (
            <button
              type="button"
              id="timelapse-play-btn"
              className={`control-pill ${isPlaying ? 'active' : ''}`}
              onClick={() => setIsPlaying(p => !p)}
              title="Play timelapse progression animation"
            >
              <span>{isPlaying ? '⏸ Pause' : '▶ Play Timelapse'}</span>
            </button>
          )}

          <div className="overlay-dropdown-group">
            <span className="control-label">Overlay:</span>
            <select
              id="overlay-style-select"
              className="temporal-select-sm"
              value={overlayStyle}
              onChange={e => setOverlayStyle(e.target.value as any)}
            >
              <option value="polygon">Exact Polygons</option>
              <option value="bbox">Bounding Box</option>
              <option value="none">Raw Image</option>
            </select>
          </div>

          <button
            type="button"
            id="toggle-card-theme"
            className="control-pill"
            onClick={() => setThemeMode(m => (m === 'lightCard' ? 'cyberDark' : 'lightCard'))}
            title="Toggle between Reference Light Card and Cyber Dark theme"
          >
            <span>{themeMode === 'lightCard' ? '🌙 Dark Frame' : '☀️ Light Frame'}</span>
          </button>
        </div>
      </div>

      {/* ── Main Comparison Container ── */}
      <div className="temporal-showcase-container">
        <section className="temporal-analyst-panel" aria-label="Geospatial and raster metadata">
          <div className="temporal-panel-heading">
            <div>
              <h3>Scene &amp; geospatial metadata</h3>
              <p>Location, footprint and source details for this temporal sample</p>
            </div>
            <span className="temporal-source-badge">{data.source}</span>
          </div>
          <div className="temporal-metadata-grid">
            <div className="temporal-metadata-item">
              <span>Representative latitude / longitude</span>
              <strong>{data.coordinates.latitude.toFixed(6)}, {data.coordinates.longitude.toFixed(6)}</strong>
            </div>
            <div className="temporal-metadata-item">
              <span>Footprint bounds (W / S / E / N)</span>
              <strong>{data.footprint.west.toFixed(5)}, {data.footprint.south.toFixed(5)}, {data.footprint.east.toFixed(5)}, {data.footprint.north.toFixed(5)}</strong>
            </div>
            <div className="temporal-metadata-item">
              <span>Footprint coordinate reference</span>
              <strong>{data.spatialReference}</strong>
            </div>
            <div className="temporal-metadata-item">
              <span>Raster asset</span>
              <strong>{data.rasterFormat}</strong>
            </div>
            <div className="temporal-metadata-item temporal-metadata-wide">
              <span>GeoTIFF / COG metadata</span>
              <strong>{data.rasterMetadataStatus}</strong>
            </div>
            <div className="temporal-metadata-item">
              <span>Acquisition dates · sensor</span>
              <strong>{baselineStage.date} → {targetStage.date} · {baselineStage.sensor ?? 'Sensor not recorded'}</strong>
            </div>
            <div className="temporal-metadata-item">
              <span>Cloud cover (baseline / target)</span>
              <strong>{baselineStage.cloudCoverPct ?? '—'}% / {targetStage.cloudCoverPct ?? '—'}%</strong>
            </div>
          </div>
        </section>

        <section className="temporal-analyst-tools" aria-label="Analyst tools">
          <div>
            <h3>Analyst tools</h3>
            <p>Area values are approximate: changed fraction is applied to the geographic footprint.</p>
          </div>
          <div className="temporal-area-metrics">
            <div><span>Footprint area</span><strong>{footprintAreaKm2.toFixed(2)} km²</strong></div>
            <div><span>Estimated changed area</span><strong>{changedAreaKm2.toFixed(2)} km²</strong></div>
          </div>
          <div className="temporal-tool-actions">
            <button type="button" className="control-pill" onClick={copyCoordinates}>{copyStatus}</button>
            <button type="button" className="control-pill" onClick={exportFootprint}>Export footprint GeoJSON</button>
          </div>
        </section>

        {/* Top Header Summary Strip */}
        <div className="temporal-header-strip">
          {/* Col 1: Location */}
          <div className="metric-strip-col">
            <div className="metric-strip-icon location-icon">
              {renderHeaderIcon('location')}
            </div>
            <div className="metric-strip-content">
              <div className="metric-strip-label">Location</div>
              <div className="metric-strip-value" title={data.locationLabel}>
                {data.locationLabel}
              </div>
            </div>
          </div>

          <div className="metric-strip-divider" />

          {/* Col 2: Time range */}
          <div className="metric-strip-col">
            <div className="metric-strip-icon calendar-icon">
              {renderHeaderIcon('calendar')}
            </div>
            <div className="metric-strip-content">
              <div className="metric-strip-label">Time range</div>
              <div className="metric-strip-value">
                {data.timeRange}
              </div>
            </div>
          </div>

          <div className="metric-strip-divider" />

          {/* Col 3: Change */}
          <div className="metric-strip-col">
            <div className="metric-strip-icon change-icon">
              {renderHeaderIcon('change')}
            </div>
            <div className="metric-strip-content">
              <div className="metric-strip-label">Change</div>
              <div className="metric-strip-value" title={data.changeType}>
                {data.changeType}
              </div>
            </div>
          </div>

          <div className="metric-strip-divider" />

          {/* Col 4: Earliest supported change */}
          <div className="metric-strip-col">
            <div className="metric-strip-icon clock-icon">
              {renderHeaderIcon('clock')}
            </div>
            <div className="metric-strip-content">
              <div className="metric-strip-label">Earliest supported change</div>
              <div className="metric-strip-value">
                {data.earliestSupportedChange}
              </div>
            </div>
          </div>

          <div className="metric-strip-divider" />

          {/* Col 5: Confidence */}
          <div className="metric-strip-col">
            <div className="metric-strip-icon confidence-icon">
              {renderHeaderIcon('shield')}
            </div>
            <div className="metric-strip-content">
              <div className="metric-strip-label">Confidence</div>
              <div className="metric-strip-value confidence-value">
                {data.confidence.toFixed(2)}
              </div>
            </div>
          </div>
        </div>

        {/* ── View Mode: Timeline Progression (Primary Reference View) ── */}
        {viewMode === 'progression' && (
          <div className="temporal-progression-row">
            {/* Stages Sequence */}
            {data.stages.map((stage: TemporalStageItem, index: number) => {
              const isCurrentActive = isPlaying && activeStep === index;
              return (
                <React.Fragment key={stage.id}>
                  <div
                    className={`temporal-card-wrapper ${isCurrentActive ? 'pulse-active' : ''}`}
                    onClick={() =>
                      setZoomedItem({
                        title: `${stage.year} Baseline Observation`,
                        subtitle: `${stage.title} • Date: ${stage.date}`,
                        imageUrl: stage.imageUrl,
                        date: stage.date,
                      })
                    }
                  >
                    {/* Upper Image Box */}
                    <div className="temporal-image-box">
                      <img
                        src={stage.imageUrl}
                        alt={`${stage.year} - ${stage.title}`}
                        className="temporal-image"
                        loading="lazy"
                      />
                      {/* Year Badge at Top-Left */}
                      <div className="temporal-year-badge">
                        {stage.year}
                      </div>
                      <div className="zoom-indicator" title="Click to enlarge">
                        🔍
                      </div>
                    </div>

                    {/* Bottom Status Chip */}
                    <div className={`temporal-chip ${getChipClass(stage.variant)}`}>
                      <span className="temporal-chip-icon">
                        {renderChipIcon(stage.iconType)}
                      </span>
                      <div className="temporal-chip-text">
                        <span className="temporal-chip-title">{stage.title}</span>
                        {stage.subtitle && (
                          <span className="temporal-chip-sub">{stage.subtitle}</span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Arrow Connector between stages */}
                  {index < data.stages.length - 1 && (
                    <div className="temporal-arrow-connector" aria-hidden="true">
                      <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M5 12h14M12 5l7 7-7 7" />
                      </svg>
                    </div>
                  )}
                </React.Fragment>
              );
            })}

            {/* Separator / Arrow before Detected Changes */}
            <div className="temporal-arrow-connector final-arrow" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </div>

            {/* Stage 4: Detected Changes Card */}
            <div
              className={`temporal-card-wrapper detected-changes-card ${
                isPlaying && activeStep === 3 ? 'pulse-active' : ''
              }`}
              onClick={() =>
                setZoomedItem({
                  title: data.detectedChanges.title,
                  subtitle: `${data.detectedChanges.subRange} • Change mask overlay`,
                  imageUrl: data.detectedChanges.imageUrl,
                  date: data.timeRange,
                  isDetected: true,
                })
              }
            >
              {/* Upper Image Box with SVG Polygon Overlay */}
              <div className="temporal-image-box">
                <img
                  src={data.detectedChanges.imageUrl}
                  alt={data.detectedChanges.title}
                  className="temporal-image"
                  loading="lazy"
                />

                {/* SVG Polygon & BBox Overlays */}
                {overlayStyle === 'polygon' && (
                  <svg
                    viewBox="0 0 100 100"
                    preserveAspectRatio="none"
                    className="temporal-mask-svg"
                    style={{ opacity: maskOpacity }}
                  >
                    <defs>
                      <filter id="red-glow" x="-20%" y="-20%" width="140%" height="140%">
                        <feDropShadow dx="0" dy="0" stdDeviation="0.8" floodColor="#ef4444" floodOpacity="0.8" />
                      </filter>
                    </defs>
                    {data.detectedChanges.polygons.map((poly, pIdx) => (
                      <polygon
                        key={pIdx}
                        points={poly.points}
                        className="change-polygon"
                        filter="url(#red-glow)"
                      >
                        <title>{poly.label || `Change zone #${pIdx + 1}`}</title>
                      </polygon>
                    ))}
                  </svg>
                )}

                {overlayStyle === 'bbox' && data.detectedChanges.boundingBoxes && (
                  <svg
                    viewBox="0 0 400 400"
                    className="temporal-mask-svg"
                    style={{ opacity: maskOpacity }}
                  >
                    {data.detectedChanges.boundingBoxes.map((box, bIdx) => (
                      <g key={bIdx}>
                        <rect
                          x={box.x}
                          y={box.y}
                          width={box.width}
                          height={box.height}
                          fill="rgba(239, 68, 68, 0.25)"
                          stroke="#ef4444"
                          strokeWidth="2.5"
                          strokeDasharray="6 3"
                        />
                        <rect
                          x={box.x}
                          y={Math.max(0, box.y - 22)}
                          width={Math.min(box.label.length * 7 + 40, 200)}
                          height="18"
                          fill="#ef4444"
                          rx="2"
                        />
                        <text
                          x={box.x + 4}
                          y={Math.max(12, box.y - 8)}
                          fill="#ffffff"
                          fontSize="9.5"
                          fontFamily="sans-serif"
                          fontWeight="700"
                        >
                          {box.label}
                        </text>
                      </g>
                    ))}
                  </svg>
                )}

                {/* Detected Changes Top-Left Badge */}
                <div className="temporal-year-badge detected-badge">
                  <div className="detected-badge-main">{data.detectedChanges.title}</div>
                  <div className="detected-badge-sub">{data.detectedChanges.subRange}</div>
                </div>

                <div className="zoom-indicator" title="Click to enlarge">
                  🔍
                </div>
              </div>

              {/* Bottom Status Chip (Red Tinted) */}
              <div className={`temporal-chip ${getChipClass(data.detectedChanges.variant)}`}>
                <span className="temporal-chip-icon">
                  {renderChipIcon(data.detectedChanges.iconType)}
                </span>
                <div className="temporal-chip-text">
                  <span className="temporal-chip-title">{data.detectedChanges.chipTitle}</span>
                  {data.detectedChanges.chipSubtitle && (
                    <span className="temporal-chip-sub">
                      {data.detectedChanges.chipSubtitle}
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── View Mode: Interactive Swipe Slider ── */}
        {viewMode === 'slider' && (
          <div className="temporal-slider-container">
            <div
              className="temporal-slider-viewport"
              onMouseMove={handleSliderDrag}
            >
              {/* After / Target Image (Base) */}
              <img
                src={targetStage.imageUrl}
                alt="Target Epoch"
                className="slider-base-image"
              />

              {/* Polygons on Target Image */}
              {overlayStyle === 'polygon' && (
                <svg
                  viewBox="0 0 100 100"
                  preserveAspectRatio="none"
                  className="temporal-mask-svg"
                  style={{ opacity: maskOpacity }}
                >
                  {data.detectedChanges.polygons.map((poly, pIdx) => (
                    <polygon
                      key={pIdx}
                      points={poly.points}
                      className="change-polygon"
                    />
                  ))}
                </svg>
              )}

              {/* Before / Baseline Image (Clipped) */}
              <div
                className="slider-clip-layer"
                style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
              >
                <img
                  src={baselineStage.imageUrl}
                  alt="Baseline Epoch"
                  className="slider-base-image"
                />
              </div>

              {/* Divider Line & Handle */}
              <div
                className="slider-divider-line"
                style={{ left: `${sliderPos}%` }}
              >
                <div className="slider-handle-pill">⇔ Drag</div>
              </div>

              <div className="slider-corner-badge badge-left">
                {baselineStage.year} ({baselineStage.date})
              </div>
              <div className="slider-corner-badge badge-right">
                {targetStage.year} ({targetStage.date})
              </div>
            </div>

            <div className="slider-instructions">
              <span>← {baselineStage.title.toUpperCase()}</span>
              <span>DRAG TO REVEAL DIFFERENCE</span>
              <span>{targetStage.title.toUpperCase()} →</span>
            </div>
          </div>
        )}

        {/* ── View Mode: Side-by-Side ── */}
        {viewMode === 'sideBySide' && (
          <div className="temporal-side-by-side-grid">
            {/* Baseline Tile */}
            <div className="side-tile-box">
              <div className="side-tile-header">
                <span className="side-tile-title">Baseline ({baselineStage.year})</span>
                <span className="side-tile-meta">{baselineStage.date}</span>
              </div>
              <div className="side-tile-img-box">
                <img
                  src={baselineStage.imageUrl}
                  alt="Baseline"
                  className="side-tile-img"
                />
              </div>
              <div className={`temporal-chip ${getChipClass(baselineStage.variant)}`} style={{ margin: '0.75rem' }}>
                <span className="temporal-chip-icon">{renderChipIcon(baselineStage.iconType)}</span>
                <span className="temporal-chip-title">{baselineStage.title}</span>
              </div>
            </div>

            {/* Target Tile with Overlays */}
            <div className="side-tile-box">
              <div className="side-tile-header">
                <span className="side-tile-title" style={{ color: '#ef4444' }}>
                  Target & Detected ({targetStage.year})
                </span>
                <span className="side-tile-meta">{targetStage.date}</span>
              </div>
              <div className="side-tile-img-box">
                <img
                  src={targetStage.imageUrl}
                  alt="Target"
                  className="side-tile-img"
                />
                {overlayStyle === 'polygon' && (
                  <svg
                    viewBox="0 0 100 100"
                    preserveAspectRatio="none"
                    className="temporal-mask-svg"
                    style={{ opacity: maskOpacity }}
                  >
                    {data.detectedChanges.polygons.map((poly, pIdx) => (
                      <polygon
                        key={pIdx}
                        points={poly.points}
                        className="change-polygon"
                      />
                    ))}
                  </svg>
                )}
              </div>
              <div className={`temporal-chip ${getChipClass(data.detectedChanges.variant)}`} style={{ margin: '0.75rem' }}>
                <span className="temporal-chip-icon">{renderChipIcon(data.detectedChanges.iconType)}</span>
                <span className="temporal-chip-title">{data.detectedChanges.chipTitle}</span>
                <span className="temporal-chip-sub">{data.detectedChanges.chipSubtitle}</span>
              </div>
            </div>
          </div>
        )}

        {/* ── Footer Analysis Note ── */}
        <div className="temporal-footer-note">
          <div className="footer-note-left">
            <span className="footer-tag">AI Temporal Analysis:</span>
            <p className="footer-text">{data.description}</p>
          </div>
          <div className="footer-note-right">
            <div className="mini-stat">
              <span className="mini-stat-label">Changed Footprint</span>
              <span className="mini-stat-val val-rose">
                {(data.changedPixelFraction * 100).toFixed(1)}%
              </span>
            </div>
            <div className="mini-stat">
              <span className="mini-stat-label">Confidence</span>
              <span className="mini-stat-val val-emerald">
                {(data.confidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Modal Lightbox for Zoomed High-Resolution Inspection ── */}
      {zoomedItem && (
        <div className="temporal-modal-backdrop" onClick={() => setZoomedItem(null)}>
          <div className="temporal-modal-content" onClick={e => e.stopPropagation()}>
            <div className="temporal-modal-header">
              <div>
                <h3 className="temporal-modal-title">{zoomedItem.title}</h3>
                {zoomedItem.subtitle && (
                  <p className="temporal-modal-subtitle">{zoomedItem.subtitle}</p>
                )}
              </div>
              <button
                type="button"
                className="modal-close-btn"
                onClick={() => setZoomedItem(null)}
              >
                ✕
              </button>
            </div>

            <div className="temporal-modal-viewport">
              <img
                src={zoomedItem.imageUrl}
                alt={zoomedItem.title}
                className="temporal-modal-image"
              />
              {zoomedItem.isDetected && overlayStyle === 'polygon' && (
                <svg
                  viewBox="0 0 100 100"
                  preserveAspectRatio="none"
                  className="temporal-mask-svg"
                  style={{ opacity: maskOpacity }}
                >
                  {data.detectedChanges.polygons.map((poly, pIdx) => (
                    <polygon
                      key={pIdx}
                      points={poly.points}
                      className="change-polygon"
                    />
                  ))}
                </svg>
              )}
            </div>

            <div className="temporal-modal-footer">
              <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                <span className="badge badge-cyan">Sensor: Sentinel-2 L2A</span>
                <span className="badge badge-emerald">Co-registered</span>
                <span className="badge badge-rose">Confidence: {(data.confidence * 100).toFixed(0)}%</span>
              </div>
              <button
                type="button"
                className="secondary-btn"
                onClick={() => setZoomedItem(null)}
              >
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
