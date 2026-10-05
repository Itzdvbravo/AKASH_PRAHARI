import { SearchResultItem } from '../../types/api.types';
import { LOCATIONS } from '../../mock/mockData';
import { resolveApiUrl } from '../../api/client';

interface SearchResultCardProps {
  item: SearchResultItem;
  onInspect: (item: SearchResultItem) => void;
}

export const SearchResultCard: React.FC<SearchResultCardProps> = ({ item, onInspect }) => {
  const confidencePct = Math.round(item.confidence.score * 100);
  const loc = LOCATIONS.find(l => l.id === item.tile_ref.location_id);

  const getBadgeClass = (score: number) => {
    if (score >= 0.95) return 'badge-emerald';
    if (score >= 0.85) return 'badge-cyan';
    return 'badge-amber';
  };

  const imgSrc = resolveApiUrl(item.thumbnail_url);

  const dateLabel = item.tile_ref.date;

  return (
    <div className="result-card" id={`result-card-${item.tile_ref.tile_id}`}>
      {/* Satellite image thumbnail with epoch toggle */}
      <div className="thumbnail-wrapper">
        <img
          src={imgSrc}
          alt={`${item.tile_ref.location_id} satellite thumbnail`}
          className="thumbnail-img"
          loading="lazy"
        />
        {/* Date watermark */}
        <div className="img-date-badge">{dateLabel}</div>
        {/* Confidence overlay */}
        <div className={`img-confidence-badge badge ${getBadgeClass(item.confidence.score)}`}>
          Score {confidencePct}%
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

        <div className="result-meta-row">
          <span>📅 {item.available_dates[0]}</span>
          <span style={{ color: 'var(--text-muted)' }}>→</span>
          <span>📅 {item.tile_ref.date}</span>
        </div>

        <div className="result-meta-row">
          <span>{item.available_dates.length} available epochs</span>
          <span>Rank #{item.rank}</span>
        </div>

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
