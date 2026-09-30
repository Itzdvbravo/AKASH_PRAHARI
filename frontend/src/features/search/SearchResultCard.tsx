import { useState } from 'react';
import { SearchResultItem } from '../../types/api.types';
import { LOCATIONS } from '../../mock/mockData';

interface SearchResultCardProps {
  item: SearchResultItem;
  onInspect: (item: SearchResultItem) => void;
}

export const SearchResultCard: React.FC<SearchResultCardProps> = ({ item, onInspect }) => {
  const [imgEpoch, setImgEpoch] = useState<'before' | 'after'>('before');

  const confidencePct = Math.round(item.confidence.score * 100);
  const loc = LOCATIONS.find(l => l.id === item.tile_ref.location_id);

  const getBadgeClass = (score: number) => {
    if (score >= 0.95) return 'badge-emerald';
    if (score >= 0.85) return 'badge-cyan';
    return 'badge-amber';
  };

  const imgSrc = loc
    ? (imgEpoch === 'before' ? loc.beforeUrl : loc.afterUrl)
    : item.thumbnail_url;

  const dateLabel = imgEpoch === 'before'
    ? (loc ? loc.date1 : item.available_dates[0])
    : (loc ? loc.date2 : item.tile_ref.date);

  return (
    <div className="result-card" id={`result-card-${item.tile_ref.tile_id}`}>
      {/* Satellite image thumbnail with epoch toggle */}
      <div className="thumbnail-wrapper">
        <img
          src={imgSrc}
          alt={`${item.tile_ref.location_id} ${imgEpoch} satellite view`}
          className="thumbnail-img"
          loading="lazy"
        />
        {/* Epoch toggle overlay */}
        <div className="epoch-toggle">
          <button
            type="button"
            className={`epoch-btn ${imgEpoch === 'before' ? 'active' : ''}`}
            onClick={() => setImgEpoch('before')}
          >T1</button>
          <button
            type="button"
            className={`epoch-btn ${imgEpoch === 'after' ? 'active' : ''}`}
            onClick={() => setImgEpoch('after')}
          >T2</button>
        </div>
        {/* Date watermark */}
        <div className="img-date-badge">{dateLabel}</div>
        {/* Confidence overlay */}
        <div className={`img-confidence-badge badge ${getBadgeClass(item.confidence.score)}`}>
          {confidencePct}% Match
        </div>
      </div>

      <div className="result-card-body">
        <div className="result-location">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '1.05rem', fontWeight: 700 }}>
              {loc?.label ?? item.tile_ref.location_id.toUpperCase()}
            </span>
            {loc && (
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{loc.country}</span>
            )}
          </div>
          <span className="badge badge-cyan" style={{ fontSize: '0.65rem' }}>
            {item.tile_ref.sensor}
          </span>
        </div>

        {loc && (
          <div style={{
            fontSize: '0.78rem',
            color: 'var(--text-secondary)',
            lineHeight: 1.5,
            borderLeft: '2px solid var(--border-accent)',
            paddingLeft: '0.6rem',
          }}>
            {loc.changeType}
          </div>
        )}

        <div className="result-meta-row">
          <span>📅 {loc?.date1 ?? item.available_dates[0]}</span>
          <span style={{ color: 'var(--text-muted)' }}>→</span>
          <span>📅 {loc?.date2 ?? item.tile_ref.date}</span>
        </div>

        {loc && (
          <div className="result-meta-row">
            <span style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>
              Changed area: ~{(loc.changedFraction * 100).toFixed(1)}% of tile
            </span>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>
              • {item.available_dates.length} epochs
            </span>
          </div>
        )}

        <button
          type="button"
          id={`inspect-btn-${item.tile_ref.tile_id}`}
          className="card-action-btn"
          onClick={() => onInspect(item)}
        >
          Inspect &amp; Compare Changes →
        </button>
      </div>
    </div>
  );
};
