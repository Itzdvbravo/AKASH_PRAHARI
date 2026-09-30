import { SearchResultItem } from '../../types/api.types';

interface SearchResultCardProps {
  item: SearchResultItem;
  onInspect: (item: SearchResultItem) => void;
}

export const SearchResultCard: React.FC<SearchResultCardProps> = ({ item, onInspect }) => {
  const confidencePct = Math.round(item.confidence.score * 100);

  const getBadgeClass = (score: number) => {
    if (score >= 0.9) return 'badge-emerald';
    if (score >= 0.8) return 'badge-cyan';
    return 'badge-amber';
  };

  return (
    <div className="result-card">
      <div className="thumbnail-wrapper">
        <img
          src={item.thumbnail_url}
          alt={`Tile ${item.tile_ref.tile_id}`}
          className="thumbnail-img"
          loading="lazy"
        />
      </div>

      <div className="result-card-body">
        <div className="result-location">
          <span>{item.tile_ref.location_id.toUpperCase()}</span>
          <span className={`badge ${getBadgeClass(item.confidence.score)}`}>
            {confidencePct}% Match
          </span>
        </div>

        <div className="result-meta-row">
          <span>📅 {item.tile_ref.date}</span>
          <span>•</span>
          <span className="badge badge-cyan">{item.tile_ref.sensor}</span>
        </div>

        <div className="result-meta-row" style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
          <span>Available dates: {item.available_dates.length} epochs</span>
        </div>

        <button
          type="button"
          className="card-action-btn"
          onClick={() => onInspect(item)}
        >
          Inspect & Compare Changes →
        </button>
      </div>
    </div>
  );
};
