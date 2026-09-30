import { useState } from 'react';
import { ComparisonResponse, ChangeDetectionResponse } from '../../types/api.types';

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
  isLoading = false
}) => {
  const [showMask, setShowMask] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [maskOpacity, setMaskOpacity] = useState(0.85);

  const before = comparison.before;
  const after = comparison.after;

  return (
    <div className="comparison-layout">
      {/* Viewer Toolbar */}
      <div className="comparison-toolbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>
            Location: {before.tile_ref.location_id.toUpperCase()}
          </span>
          <span className="badge badge-cyan">{before.tile_ref.sensor}</span>
          {comparison.coregistered && (
            <span className="badge badge-emerald">Coregistered</span>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <button
            type="button"
            className={`control-pill ${showMask ? 'active' : ''}`}
            onClick={() => setShowMask(!showMask)}
          >
            <span>{showMask ? '✓' : '○'}</span>
            <span>Change Mask</span>
          </button>

          <button
            type="button"
            className={`control-pill ${showBoxes ? 'active' : ''}`}
            onClick={() => setShowBoxes(!showBoxes)}
          >
            <span>{showBoxes ? '✓' : '○'}</span>
            <span>Bounding Box</span>
          </button>

          {showMask && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              <span>Opacity:</span>
              <input
                type="range"
                min="0.2"
                max="1.0"
                step="0.05"
                value={maskOpacity}
                onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
                style={{ width: '80px', cursor: 'pointer' }}
                aria-label="Mask opacity"
              />
              <span>{Math.round(maskOpacity * 100)}%</span>
            </div>
          )}

          <button
            type="button"
            className="secondary-btn"
            onClick={onRefreshDetection}
            disabled={isLoading}
          >
            {isLoading ? 'Processing...' : 'Re-run Detection'}
          </button>
        </div>
      </div>

      {/* Side-by-Side Satellite Comparison */}
      <div className="viewer-grid">
        {/* Before Tile */}
        <div className="tile-box">
          <div className="tile-box-header">
            <span>PRE-EVENT BASELINE</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              {before.tile_ref.date}
            </span>
          </div>
          <div className="tile-viewport">
            <img
              src={before.image_url}
              alt="Baseline Tile"
              className="tile-image"
            />
          </div>
          <div style={{ padding: '0.65rem 1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Cloud cover: {before.cloud_cover_pct ?? 0}% • Tile: {before.tile_ref.tile_id}
          </div>
        </div>

        {/* After Tile */}
        <div className="tile-box">
          <div className="tile-box-header">
            <span style={{ color: 'var(--accent-cyan)' }}>POST-EVENT TARGET</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              {after.tile_ref.date}
            </span>
          </div>
          <div className="tile-viewport">
            <img
              src={after.image_url}
              alt="Post-Event Target Tile"
              className="tile-image"
            />

            {/* Change Mask Overlay */}
            {showMask && changeDetection && (
              <img
                src={changeDetection.mask_url}
                alt="Change Detection Mask Overlay"
                className="mask-overlay"
                style={{ opacity: maskOpacity }}
              />
            )}

            {/* Bounding Box Overlay */}
            {showBoxes && changeDetection && (
              <svg
                viewBox="0 0 400 400"
                className="bbox-overlay"
              >
                {changeDetection.bounding_boxes.map((box, idx) => (
                  <g key={idx}>
                    <rect
                      x={box.x}
                      y={box.y}
                      width={box.width}
                      height={box.height}
                      fill="none"
                      stroke="#f43f5e"
                      strokeWidth="2.5"
                      strokeDasharray="4 2"
                    />
                    <rect
                      x={box.x}
                      y={box.y - 20}
                      width="180"
                      height="18"
                      fill="#111827"
                      fillOpacity="0.9"
                      rx="3"
                    />
                    <text
                      x={box.x + 5}
                      y={box.y - 7}
                      fill="#f43f5e"
                      fontSize="10"
                      fontFamily="sans-serif"
                      fontWeight="600"
                    >
                      {box.label.slice(0, 22)} ({Math.round(box.confidence.score * 100)}%)
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
    </div>
  );
};
