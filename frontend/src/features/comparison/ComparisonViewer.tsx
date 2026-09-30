import { useState } from 'react';
import { ComparisonResponse, ChangeDetectionResponse } from '../../types/api.types';
import { LOCATIONS } from '../../mock/mockData';

interface ComparisonViewerProps {
  comparison: ComparisonResponse;
  changeDetection: ChangeDetectionResponse | null;
  onRefreshDetection: () => void;
  isLoading?: boolean;
}

export const ComparisonViewer: React.FC<ComparisonViewerProps> = ({
  comparison,
  changeDetection,
  onRefreshDetection,
  isLoading = false,
}) => {
  const [showBoxes, setShowBoxes] = useState(true);
  const [sliderMode, setSliderMode] = useState(false);
  const [sliderPos, setSliderPos] = useState(50);

  const before = comparison.before;
  const after = comparison.after;
  const loc = LOCATIONS.find(l => l.id === before.tile_ref.location_id);

  // Fake SVG change-mask overlay — drawn over the after image
  const maskSvg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100%" height="100%" preserveAspectRatio="none">
      <rect x="12" y="10" width="28" height="26" rx="1" fill="#f43f5e" fill-opacity="0.5" stroke="#f43f5e" stroke-width="0.8"/>
      <rect x="48" y="42" width="20" height="18" rx="1" fill="#f43f5e" fill-opacity="0.4" stroke="#f43f5e" stroke-width="0.8"/>
      <rect x="70" y="60" width="16" height="14" rx="1" fill="#f43f5e" fill-opacity="0.35" stroke="#f97316" stroke-width="0.6"/>
      <rect x="30" y="65" width="12" height="10" rx="1" fill="#fb923c" fill-opacity="0.3" stroke="#fb923c" stroke-width="0.5"/>
    </svg>
  `;
  const maskDataUrl = `data:image/svg+xml;utf8,${encodeURIComponent(maskSvg.trim())}`;

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
          {loc && (
            <span className="badge badge-amber">{loc.changeType}</span>
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
            id="toggle-bboxes"
            className={`control-pill ${showBoxes ? 'active' : ''}`}
            onClick={() => setShowBoxes(!showBoxes)}
          >
            <span>{showBoxes ? '✓' : '○'}</span>
            <span>Change Boxes</span>
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
              src={after.image_url}
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
                src={before.image_url}
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
        /* ── Side-by-side Mode ── */
        <div className="viewer-grid">
          {/* Before */}
          <div className="tile-box">
            <div className="tile-box-header">
              <span>PRE-EVENT BASELINE</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                {before.tile_ref.date}
              </span>
            </div>
            <div className="tile-viewport">
              <img src={before.image_url} alt="Baseline Tile" className="tile-image" />
            </div>
            <div style={{ padding: '0.65rem 1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Cloud cover: {before.cloud_cover_pct ?? 0}% • Tile: {before.tile_ref.tile_id}
            </div>
          </div>

          {/* After + overlays */}
          <div className="tile-box">
            <div className="tile-box-header">
              <span style={{ color: 'var(--accent-cyan)' }}>POST-EVENT TARGET</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                {after.tile_ref.date}
              </span>
            </div>
            <div className="tile-viewport">
              <img src={after.image_url} alt="Post-Event Target Tile" className="tile-image" />

              {/* Change Mask Overlay */}
              {changeDetection && (
                <img
                  src={maskDataUrl}
                  alt="Change Detection Mask"
                  className="mask-overlay"
                  style={{ opacity: 0.82 }}
                />
              )}

              {/* Bounding Boxes */}
              {showBoxes && changeDetection && (
                <svg viewBox="0 0 400 400" className="bbox-overlay">
                  {changeDetection.bounding_boxes.map((box, idx) => (
                    <g key={idx}>
                      <rect
                        x={box.x}
                        y={box.y}
                        width={box.width}
                        height={box.height}
                        fill="none"
                        stroke={idx === 0 ? '#f43f5e' : '#fb923c'}
                        strokeWidth="2.5"
                        strokeDasharray="6 3"
                      />
                      <rect
                        x={box.x}
                        y={box.y - 22}
                        width={Math.min(box.label.length * 7.5 + 60, 240)}
                        height="19"
                        fill={idx === 0 ? '#f43f5e' : '#fb923c'}
                        fillOpacity="0.9"
                        rx="3"
                      />
                      <text
                        x={box.x + 5}
                        y={box.y - 8}
                        fill="#fff"
                        fontSize="10"
                        fontFamily="sans-serif"
                        fontWeight="700"
                      >
                        {box.label.slice(0, 28)} ({Math.round(box.confidence.score * 100)}%)
                      </text>
                    </g>
                  ))}
                </svg>
              )}
            </div>
            <div style={{ padding: '0.65rem 1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Cloud cover: {after.cloud_cover_pct ?? 0}% • Tile: {after.tile_ref.tile_id}
            </div>
          </div>
        </div>
      )}

      {/* Location description */}
      {loc && (
        <div className="ui-card" style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start' }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.35rem' }}>
              AI Interpretation
            </div>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              {loc.changeDesc}
            </p>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', minWidth: '140px' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Changed Area</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#f43f5e', fontFamily: 'var(--font-mono)' }}>
              {(loc.changedFraction * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginTop: '0.25rem' }}>Confidence</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--accent-emerald)', fontFamily: 'var(--font-mono)' }}>
              {(loc.confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
