import { useState } from 'react';
import { Sensor } from '../../types/api.types';

interface SearchBarProps {
  onSearch: (query: string, sensor: Sensor, location: string) => void;
  isLoading: boolean;
}

export const SearchBar: React.FC<SearchBarProps> = ({ onSearch, isLoading }) => {
  const [query, setQuery] = useState('');
  const [sensor, setSensor] = useState<Sensor>('any');
  const [location, setLocation] = useState('all');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch(query, sensor, location);
  };

  const handleQuickTag = (tagQuery: string, loc: string) => {
    setQuery(tagQuery);
    setLocation(loc);
    onSearch(tagQuery, sensor, loc);
  };

  return (
    <div className="search-container">
      <form onSubmit={handleSubmit} className="search-input-group">
        <span className="search-icon">🔎</span>
        <input
          type="text"
          className="search-input"
          placeholder="Semantic query (e.g. 'urban expansion', 'river meander', 'forest edge')..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit" className="primary-btn" disabled={isLoading}>
          {isLoading ? 'Retrieving...' : 'Search Tiles'}
        </button>
      </form>

      <div className="filter-row">
        <span className="filter-label">Filters:</span>

        <select
          className="filter-select"
          value={location}
          onChange={(e) => {
            setLocation(e.target.value);
            onSearch(query, sensor, e.target.value);
          }}
          aria-label="Filter by Location"
        >
          <option value="all">All Locations</option>
          <option value="paris">Paris, France</option>
          <option value="berlin">Berlin, Germany</option>
        </select>

        <select
          className="filter-select"
          value={sensor}
          onChange={(e) => {
            const nextSensor = e.target.value as Sensor;
            setSensor(nextSensor);
            onSearch(query, nextSensor, location);
          }}
          aria-label="Filter by Sensor"
        >
          <option value="any">All Sensors</option>
          <option value="sentinel-2">Sentinel-2 (Optical)</option>
          <option value="sentinel-1">Sentinel-1 (SAR)</option>
          <option value="landsat">Landsat-8/9</option>
        </select>

        <span className="filter-label" style={{ marginLeft: 'auto' }}>Quick Presets:</span>
        <button
          type="button"
          className="quick-tag"
          onClick={() => handleQuickTag('urban expansion', 'paris')}
        >
          Paris Urban Expansion
        </button>
        <button
          type="button"
          className="quick-tag"
          onClick={() => handleQuickTag('infrastructure', 'berlin')}
        >
          Berlin Infrastructure
        </button>
      </div>
    </div>
  );
};
