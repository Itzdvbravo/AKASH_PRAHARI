import { useEffect, useRef, useState } from 'react';
import { ComparisonResponse, ChangeDetectionResponse, GeoBBox } from '../../types/api.types';
import { LOCATIONS } from '../../mock/mockData';
import { resolveApiUrl } from '../../api/client';

interface ComparisonViewerProps {
  comparison: ComparisonResponse;
  geoBBox?: GeoBBox;
  changeDetection: ChangeDetectionResponse | null;
  onRefreshDetection: () => void;
  isLoading?: boolean;
}

export const ComparisonViewer: React.FC<ComparisonViewerProps> = ({
  comparison,
  geoBBox,
  changeDetection,
  onRefreshDetection,
  isLoading = false,
}) => {
  const [sliderMode, setSliderMode] = useState(false);
  const [sliderPos, setSliderPos] = useState(50);
  const [selectedZone, setSelectedZone] = useState<number | null>(null);
  const [selectedSemanticBox, setSelectedSemanticBox] = useState<{ x: number; y: number; width: number; height: number } | null>(null);
  const [zooms, setZooms] = useState([{ scale: 1, x: 50, y: 50 }, { scale: 1, x: 50, y: 50 }, { scale: 1, x: 50, y: 50 }]);
  const maskedViewportRef = useRef<HTMLDivElement>(null);

  const updateZoom = (index: number, scale: number, x?: number, y?: number) => {
    setZooms(current => current.map((zoom, i) => i === index
      ? { scale: Math.max(1, Math.min(5, scale)), x: x ?? zoom.x, y: y ?? zoom.y }
      : zoom));
  };

  // React delegates wheel events at the root; attach non-passive handlers directly
  // to the image viewports so browsers always honor preventDefault().
  useEffect(() => {
    const viewports = Array.from(document.querySelectorAll<HTMLElement>('.comparison-layout .tile-viewport'));
    const listeners = viewports.map((viewport, index) => {
      const listener = (event: WheelEvent) => {
        event.preventDefault();
        event.stopPropagation();
        const rect = viewport.getBoundingClientRect();
        const x = ((event.clientX - rect.left) / rect.width) * 100;
        const y = ((event.clientY - rect.top) / rect.height) * 100;
        setZooms(current => current.map((zoom, i) => i === index
          ? { ...zoom, scale: Math.max(1, Math.min(5, zoom.scale + (event.deltaY < 0 ? 0.25 : -0.25))), x, y }
          : zoom));
      };
      viewport.addEventListener('wheel', listener, { passive: false });
      return () => viewport.removeEventListener('wheel', listener);
    });
    return () => listeners.forEach(remove => remove());
  }, [sliderMode, changeDetection]);

  const zoomedImageStyle = (index: number): React.CSSProperties => ({
    transform: `translate(${zooms[index].scale > 1 ? 50 - zooms[index].x : 0}%, ${zooms[index].scale > 1 ? 50 - zooms[index].y : 0}%) scale(${zooms[index].scale})`,
    transformOrigin: `${zooms[index].x}% ${zooms[index].y}%`,
    transition: 'transform 180ms ease-out',
  });

  const focusDetectedZone = (index: number, box: { x: number; y: number; width: number; height: number }) => {
    setSelectedZone(index);
    setSelectedSemanticBox(null);
    updateZoom(2, 3, ((box.x + box.width / 2) / 256) * 100, ((box.y + box.height / 2) / 256) * 100);
    maskedViewportRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  useEffect(() => {
    setSelectedZone(null);
    setSelectedSemanticBox(null);
  }, [changeDetection]);

  const before = comparison.before;
  const after = comparison.after;
  const loc = LOCATIONS.find(l => l.id === before.tile_ref.location_id);
  const hasGeoBBox = Boolean(geoBBox && !(geoBBox.west === 0 && geoBBox.south === 0 && geoBBox.east === 0.05 && geoBBox.north === 0.05));
  const coordinates = geoBBox && hasGeoBBox
    ? { latitude: (geoBBox.south + geoBBox.north) / 2, longitude: (geoBBox.west + geoBBox.east) / 2 }
    : null;
  const footprintAreaKm2 = geoBBox && hasGeoBBox
    ? (6371.0088 ** 2) * (Math.PI / 180) * (geoBBox.east - geoBBox.west) *
      (Math.sin(geoBBox.north * Math.PI / 180) - Math.sin(geoBBox.south * Math.PI / 180))
    : null;
  const [copyStatus, setCopyStatus] = useState('Copy coordinates');

  const copyCoordinates = async () => {
    if (!coordinates) return;
    try {
      await navigator.clipboard.writeText(`${coordinates.latitude.toFixed(6)}, ${coordinates.longitude.toFixed(6)}`);
      setCopyStatus('Coordinates copied');
      window.setTimeout(() => setCopyStatus('Copy coordinates'), 1800);
    } catch {
      setCopyStatus('Clipboard unavailable');
      window.setTimeout(() => setCopyStatus('Copy coordinates'), 1800);
    }
  };

  const exportFootprint = () => {
    if (!geoBBox || !hasGeoBBox) return;
    const feature = {
      type: 'Feature',
      properties: {
        location: loc ? `${loc.label}, ${loc.country}` : before.tile_ref.location_id,
        tile_id: before.tile_ref.tile_id,
        date_before: before.tile_ref.date,
        date_after: after.tile_ref.date,
        sensor: before.tile_ref.sensor,
        center_latitude: coordinates?.latitude,
        center_longitude: coordinates?.longitude,
        bbox_west_south_east_north: [geoBBox.west, geoBBox.south, geoBBox.east, geoBBox.north],
        footprint_area_estimate_km2: footprintAreaKm2 === null ? null : Number(footprintAreaKm2.toFixed(3)),
        changed_ground_area_km2: null,
        changed_pixel_fraction: changeDetection?.summary.changed_pixel_fraction ?? null,
        detector: changeDetection?.summary.detector ?? null,
        source: changeDetection?.summary.source_provenance ?? 'DynamicEarthNet PlanetFusion archive',
        display_asset: 'Rendered 3-band RGB PNG preview',
        source_rasters: 'DynamicEarthNet PlanetFusion RGB composites and monthly categorical land-cover labels',
        geotiff_georeferencing: 'DynamicEarthNet STAC projection metadata; 3 m GSD',
        exported_geometry_note: 'Approximate indexed city bounding box, not a precise tile footprint',
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[[geoBBox.west, geoBBox.south], [geoBBox.east, geoBBox.south], [geoBBox.east, geoBBox.north], [geoBBox.west, geoBBox.north], [geoBBox.west, geoBBox.south]]],
      },
    };
    const blob = new Blob([JSON.stringify({ type: 'FeatureCollection', features: [feature] }, null, 2)], { type: 'application/geo+json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${before.tile_ref.location_id}-${before.tile_ref.tile_id}-change-footprint.geojson`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const detectorZones = (changeDetection?.bounding_boxes ?? []).map(box => ({
    x: box.x, y: box.y, width: box.width, height: box.height,
    label: box.label.replace(/_/g, ' '),
    confidence: box.confidence.score,
  }));
  const displayZones = detectorZones;
  const maskUrl = changeDetection ? `${resolveApiUrl(changeDetection.mask_url)}?overlay=true` : undefined;
  const semanticMaskUrl = changeDetection?.semantic_mask_url
    ? `${resolveApiUrl(changeDetection.semantic_mask_url)}?semantic_overlay=true`
    : undefined;
  const beforeImageUrl = before.image_url.startsWith('/api/') ? resolveApiUrl(before.image_url) : before.image_url;
  const afterImageUrl = after.image_url.startsWith('/api/') ? resolveApiUrl(after.image_url) : after.image_url;

  const handleSliderDrag = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!sliderMode) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    setSliderPos(Math.max(5, Math.min(95, (x / rect.width) * 100)));
  };

  return (
    <div className="comparison-layout">
      {/* Toolbar */}
      <div className="comparison-toolbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>
            {loc ? `${loc.label}, ${loc.country}` : before.tile_ref.location_id.toUpperCase()}
          </span>
          <span className="badge badge-cyan">{before.tile_ref.sensor}</span>
          {comparison.coregistered && (
            <span className="badge badge-emerald">Co-registered</span>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <button
            type="button"
            id="toggle-slider-mode"
            className={`control-pill ${sliderMode ? 'active' : ''}`}
            onClick={() => setSliderMode(s => !s)}
          >
            <span>{sliderMode ? '⊣' : '⊢'}</span>
            <span>{sliderMode ? 'Slider On' : 'Slider Mode'}</span>
          </button>

          <button
            type="button"
            id="rerun-detection-btn"
            className="secondary-btn"
            onClick={onRefreshDetection}
            disabled={isLoading}
          >
            {isLoading ? (
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <span className="spinner" /> Processing…
              </span>
            ) : '⟳ Re-run Detection'}
          </button>
        </div>
      </div>
      
      {sliderMode ? (
        /* ── Swipe Slider Mode ── */
        <div className="ui-card" style={{ padding: 0, overflow: 'hidden' }}>
          <div
            style={{ position: 'relative', width: '100%', aspectRatio: '16/9', cursor: 'ew-resize', userSelect: 'none' }}
            onMouseMove={handleSliderDrag}
          >
            {/* After (base) */}
            <img
              src={afterImageUrl}
              alt="After"
              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }}
            />
            {/* Before (clipped) */}
            <div style={{
              position: 'absolute', inset: 0,
              clipPath: `inset(0 ${100 - sliderPos}% 0 0)`,
              transition: 'clip-path 0.02s',
            }}>
              <img
                src={beforeImageUrl}
                alt="Before"
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
            </div>
            {/* Divider */}
            <div style={{
              position: 'absolute', top: 0, bottom: 0, left: `${sliderPos}%`,
              width: '2px', background: 'white', zIndex: 5,
              transform: 'translateX(-50%)',
              boxShadow: '0 0 8px rgba(0,0,0,0.8)',
            }}>
              <div style={{
                position: 'absolute', top: '50%', left: '50%',
                transform: 'translate(-50%, -50%)',
                width: 28, height: 28, borderRadius: '50%',
                background: 'white', boxShadow: '0 2px 8px rgba(0,0,0,0.5)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '12px', color: '#111',
              }}>⇔</div>
            </div>
            <div className="img-date-badge" style={{ left: '0.75rem', bottom: '0.75rem' }}>{before.tile_ref.date}</div>
            <div className="img-date-badge" style={{ right: '0.75rem', bottom: '0.75rem', left: 'auto' }}>{after.tile_ref.date}</div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            <span>← PRE-EVENT BASELINE ({before.tile_ref.date})</span>
            <span>POST-EVENT TARGET ({after.tile_ref.date}) →</span>
          </div>
        </div>
      ) : (
        /* Before, after, and masked views */
        <div className="viewer-grid">
          {/* Before */}
          <div className="tile-box">
            <div className="tile-box-header">
              <span>BEFORE</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                {before.tile_ref.date}
              </span>
            </div>
            <div className="tile-viewport">
              <img src={beforeImageUrl} alt="Baseline Tile" className="tile-image" style={zoomedImageStyle(0)} />
              <ZoomControls scale={zooms[0].scale} onChange={scale => updateZoom(0, scale)} onReset={() => updateZoom(0, 1, 50, 50)} />
            </div>
            <div style={{ padding: '0.65rem 1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Cloud cover: {before.cloud_cover_pct === null ? 'N/A' : `${before.cloud_cover_pct}%`} • Tile: {before.tile_ref.tile_id}
            </div>
          </div>

          {/* Clean after image */}
          <div className="tile-box">
            <div className="tile-box-header">
              <span style={{ color: 'var(--accent-cyan)' }}>AFTER</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                {after.tile_ref.date}
              </span>
            </div>
            <div className="tile-viewport">
              <img src={afterImageUrl} alt="Post-Event Target Tile" className="tile-image" style={zoomedImageStyle(1)} />
              <ZoomControls scale={zooms[1].scale} onChange={scale => updateZoom(1, scale)} onReset={() => updateZoom(1, 1, 50, 50)} />
            </div>
            <div style={{ padding: '0.65rem 1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Cloud cover: {after.cloud_cover_pct === null ? 'N/A' : `${after.cloud_cover_pct}%`} • Tile: {after.tile_ref.tile_id}
            </div>
          </div>

          {/* Masked after image */}
          <div className="tile-box">
            <div className="tile-box-header">
              <span style={{ color: 'var(--accent-rose)' }}>MASKED REGION</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                {after.tile_ref.date}
              </span>
            </div>
            <div className="tile-viewport" ref={maskedViewportRef}>
              {changeDetection ? (
                <>
                  <img src={afterImageUrl} alt="Post-event satellite image" className="tile-image" style={zoomedImageStyle(2)} />
                  <img
                    src={maskUrl}
                    alt="Image detector predicted binary change mask"
                    className="mask-overlay"
                    style={{ opacity: semanticMaskUrl ? 0.28 : 1, ...zoomedImageStyle(2) }}
                  />
                  {semanticMaskUrl && (
                    <img
                      src={semanticMaskUrl}
                      alt="Model-predicted land-cover transition overlay"
                      className="mask-overlay"
                      style={{ opacity: 1, ...zoomedImageStyle(2) }}
                    />
                  )}
                  {selectedZone !== null && displayZones[selectedZone] && (
                    <svg className="selected-zone-overlay" viewBox="0 0 256 256" preserveAspectRatio="none" style={zoomedImageStyle(2)} aria-hidden="true">
                      <rect
                        x={displayZones[selectedZone].x}
                        y={displayZones[selectedZone].y}
                        width={displayZones[selectedZone].width}
                        height={displayZones[selectedZone].height}
                        rx="2"
                        className="selected-zone-mark"
                      />
                    </svg>
                  )}
                  {selectedSemanticBox && (
                    <svg className="selected-zone-overlay" viewBox="0 0 256 256" preserveAspectRatio="none" style={zoomedImageStyle(2)} aria-hidden="true">
                      <rect
                        x={selectedSemanticBox.x}
                        y={selectedSemanticBox.y}
                        width={selectedSemanticBox.width}
                        height={selectedSemanticBox.height}
                        rx="2"
                        className="selected-zone-mark"
                      />
                    </svg>
                  )}
                  <svg className="detected-zone-hit-overlay" viewBox="0 0 256 256" preserveAspectRatio="none" style={zoomedImageStyle(2)} aria-label="Model-predicted regions; select a region to zoom">
                    {displayZones.map((box, index) => (
                      <rect
                        key={`zone-hit-${index}`}
                        x={box.x}
                        y={box.y}
                        width={box.width}
                        height={box.height}
                        fill="rgba(0,0,0,0)"
                        className="detected-zone-hit-target"
                        role="button"
                        tabIndex={0}
                        aria-label={`Zoom to ${box.label} region ${index + 1}`}
                        onClick={() => focusDetectedZone(index, box)}
                        onKeyDown={event => {
                          if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault();
                            focusDetectedZone(index, box);
                          }
                        }}
                      />
                    ))}
                  </svg>
                </>
              ) : (
                <div className="mask-placeholder">Run change detection to view the mask.</div>
              )}
              <ZoomControls scale={zooms[2].scale} onChange={scale => updateZoom(2, scale)} onReset={() => updateZoom(2, 1, 50, 50)} />
            </div>
            <div style={{ padding: '0.65rem 1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {semanticMaskUrl
                ? 'Colored pixels show predicted land-cover transitions; the faint red overlay shows all predicted binary changes.'
                : 'Red overlay marks pixels predicted as changed by the image detector.'}
            </div>
          </div>
        </div>
      )}

      {!sliderMode && changeDetection && (
        <div className="detection-notes" aria-label="Detected changes">
          <h3 className="section-title detected-regions-heading">Model-predicted change regions</h3>
          {displayZones.length > 0 ? (
            <ul>
              {displayZones.map((zone, idx) => (
                <li key={`${zone.label}-${idx}`} className={selectedZone === idx ? 'selected-detected-zone' : ''}>
                  <button type="button" className="detected-zone-button" aria-pressed={selectedZone === idx} onClick={() => focusDetectedZone(idx, zone)}>
                    <strong>{zone.label}</strong> at ({zone.x}, {zone.y}), {zone.width} × {zone.height} px
                    {zone.confidence !== undefined && (
                      <span className="region-confidence-label"> · {Math.round(zone.confidence * 100)}% model confidence</span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p>No individual regions were predicted by the image detector.</p>
          )}
          {changeDetection.semantic_mask_url && (
            <div className="semantic-transition-results" aria-label="Model-predicted semantic transitions">
              <h3 className="section-title">Predicted land-cover transitions</h3>
              {(changeDetection.semantic_transitions ?? []).length ? (
                <ul>
                  {(changeDetection.semantic_transitions ?? []).map((transition, index) => (
                  <li key={`${transition.from_class}-${transition.to_class}-${index}`}>
                    <button
                      type="button"
                      className="semantic-transition-button"
                      disabled={!transition.bounding_boxes?.length}
                      onClick={() => {
                        const box = transition.bounding_boxes?.[0];
                        if (!box) return;
                        setSelectedZone(null);
                        setSelectedSemanticBox(box);
                        updateZoom(2, 3, ((box.x + box.width / 2) / 256) * 100, ((box.y + box.height / 2) / 256) * 100);
                        maskedViewportRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
                      }}
                    >
                      <span className="semantic-transition-swatch" style={{ backgroundColor: transition.color }} aria-hidden="true" />
                      <strong>{transition.from_class} → {transition.to_class}</strong>
                      <span>{transition.pixel_count.toLocaleString()} px · {(transition.area_fraction * 100).toFixed(2)}% of tile</span>
                    </button>
                  </li>
                  ))}
                </ul>
              ) : (
                <p>No changed land-cover transitions were predicted.</p>
              )}
            </div>
          )}
        </div>
      )}

      <section className="comparison-analyst-panel" aria-label="Scene geospatial metadata and analyst tools">
        <div className="comparison-analyst-heading">
          <div>
            <h3>Scene metadata &amp; analyst tools</h3>
            <p>Indexed scene footprint and raster details available to this comparison</p>
          </div>
            <span className="comparison-metadata-badge">{before.tile_ref.sensor} · {before.tile_ref.tile_id}</span>
        </div>
        <div className="comparison-metadata-grid">
          <div><span>Approximate location coordinate (lat / lon)</span><strong>{coordinates ? `${coordinates.latitude.toFixed(6)}, ${coordinates.longitude.toFixed(6)}` : 'Coordinates unavailable in the index'}</strong></div>
          <div><span>Approximate city bounds (W / S / E / N)</span><strong>{geoBBox && hasGeoBBox ? `${geoBBox.west.toFixed(5)}, ${geoBBox.south.toFixed(5)}, ${geoBBox.east.toFixed(5)}, ${geoBBox.north.toFixed(5)}` : 'Not provided'}</strong></div>
          <div><span>Bounds coordinate reference</span><strong>{hasGeoBBox ? 'WGS 84 longitude / latitude (EPSG:4326)' : 'Unavailable'}</strong></div>
          <div><span>Acquisitions · co-registration</span><strong>{before.tile_ref.date} → {after.tile_ref.date} · {comparison.coregistered ? 'aligned' : 'not confirmed'}</strong></div>
          <div><span>Dataset</span><strong>DynamicEarthNet-video · PlanetFusion</strong></div>
          <div><span>Displayed image / tile</span><strong>3-band RGB preview · 256 × 256 px · 3 m GSD</strong></div>
          <div><span>Georeferencing</span><strong>Dataset STAC CRS and affine transform</strong></div>
          <div><span>Change model output</span><strong>{changeDetection?.semantic_mask_url ? '49-class DynamicEarthNet semantic transitions' : 'Binary mask · semantic model not active'}</strong></div>
        </div>
        <div className="comparison-analyst-actions">
          <div className="comparison-area-metric">
            <span>Approximate city-bounds area</span>
            <strong>{footprintAreaKm2 === null ? 'Unavailable' : `${footprintAreaKm2.toFixed(2)} km²`}</strong>
          </div>
          <div className="comparison-area-metric">
            <span>Changed ground area</span>
            <strong>Unavailable without georeferencing</strong>
          </div>
          <p>The indexed bounds are city-level approximations. GeoJSON export preserves that approximate extent; the change mask cannot be placed at precise ground coordinates.</p>
          <div className="comparison-action-buttons">
            <button type="button" className="control-pill" onClick={copyCoordinates} disabled={!coordinates}>{copyStatus}</button>
            <button type="button" className="control-pill" onClick={exportFootprint} disabled={!geoBBox || !hasGeoBBox}>Export approximate bounds GeoJSON</button>
          </div>
        </div>
      </section>
    </div>
  );
};

const ZoomControls: React.FC<{ scale: number; onChange: (scale: number) => void; onReset: () => void }> = ({ scale, onChange, onReset }) => (
  <div className="tile-zoom-controls" aria-label="Image zoom controls" onWheel={event => { event.preventDefault(); event.stopPropagation(); }}>
    <button type="button" aria-label="Zoom in" onClick={() => onChange(scale + 0.5)}>+</button>
    <span>{Math.round(scale * 100)}%</span>
    <button type="button" aria-label="Zoom out" onClick={() => onChange(scale - 0.5)}>−</button>
    <button type="button" aria-label="Reset zoom" onClick={onReset}>Reset</button>
  </div>
);
