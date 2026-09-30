import { useState, useRef, useEffect } from 'react';
import { QUERY_PRESETS, QueryPreset } from '../../mock/mockData';

interface SearchBarProps {
  onSearch: (query: string, presetId: string) => void;
  isLoading: boolean;
}

const CATEGORY_ICONS: Record<string, string> = {
  construction: '🏗️',
  disaster: '⚠️',
  deforestation: '🌳',
  infrastructure: '🛣️',
  all: '🌍',
};

const CATEGORY_COLORS: Record<string, string> = {
  construction: '#06b6d4',
  disaster: '#f43f5e',
  deforestation: '#10b981',
  infrastructure: '#f59e0b',
  all: '#8b5cf6',
};

export const SearchBar: React.FC<SearchBarProps> = ({ onSearch, isLoading }) => {
  const [query, setQuery] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);
  const [highlightIdx, setHighlightIdx] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Filter presets (skip the empty "all" preset in suggestions)
  const visiblePresets = QUERY_PRESETS.filter(p => p.query);
  const filtered = query.trim()
    ? visiblePresets.filter(p => p.query.toLowerCase().includes(query.toLowerCase()))
    : visiblePresets;

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setShowDropdown(false);
    onSearch(query, '');
  };

  const selectPreset = (preset: QueryPreset) => {
    setQuery(preset.query);
    setShowDropdown(false);
    onSearch(preset.query, preset.id);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!showDropdown) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setHighlightIdx(i => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlightIdx(i => Math.max(i - 1, -1));
    } else if (e.key === 'Enter' && highlightIdx >= 0) {
      e.preventDefault();
      selectPreset(filtered[highlightIdx]);
    } else if (e.key === 'Escape') {
      setShowDropdown(false);
    }
  };

  const clearQuery = () => {
    setQuery('');
    setShowDropdown(true);
    inputRef.current?.focus();
    onSearch('', 'q_all');
  };

  return (
    <div className="search-container" ref={containerRef}>
      {/* Main search form */}
      <form onSubmit={handleSubmit} className="search-input-group" style={{ position: 'relative' }}>
        <span className="search-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
        </span>
        <input
          ref={inputRef}
          type="text"
          id="nlp-search-input"
          className="search-input"
          placeholder='Ask in natural language — e.g. "Show me where new buildings appeared between 2015–2018 in Dubai"'
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setShowDropdown(true);
            setHighlightIdx(-1);
          }}
          onFocus={() => setShowDropdown(true)}
          onKeyDown={handleKeyDown}
          autoComplete="off"
        />
        {query && (
          <button
            type="button"
            onClick={clearQuery}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              padding: '0 0.25rem',
              fontSize: '1.1rem',
              lineHeight: 1,
            }}
            aria-label="Clear search"
          >×</button>
        )}
        <button type="submit" className="primary-btn" disabled={isLoading} id="search-submit-btn">
          {isLoading ? (
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="spinner" />
              Analysing…
            </span>
          ) : 'Analyse'}
        </button>

        {/* Autocomplete dropdown */}
        {showDropdown && filtered.length > 0 && (
          <div className="search-dropdown" id="search-dropdown">
            <div className="search-dropdown-header">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
              Suggested Queries
            </div>
            {filtered.map((preset, idx) => (
              <button
                type="button"
                key={preset.id}
                className={`search-dropdown-item ${idx === highlightIdx ? 'highlighted' : ''}`}
                onClick={() => selectPreset(preset)}
                onMouseEnter={() => setHighlightIdx(idx)}
              >
                <span className="dropdown-icon">{CATEGORY_ICONS[preset.category]}</span>
                <span className="dropdown-query">{preset.query}</span>
                <span
                  className="dropdown-category"
                  style={{ color: CATEGORY_COLORS[preset.category] }}
                >
                  {preset.category}
                </span>
              </button>
            ))}
          </div>
        )}
      </form>

      {/* Quick preset chips */}
      <div className="filter-row">
        <span className="filter-label">Quick Presets:</span>
        {QUERY_PRESETS.filter(p => p.query).map(preset => (
          <button
            key={preset.id}
            type="button"
            className={`quick-tag ${query === preset.query ? 'active' : ''}`}
            onClick={() => selectPreset(preset)}
            title={preset.query}
          >
            <span>{CATEGORY_ICONS[preset.category]}</span>
            <span>{preset.query.length > 38 ? preset.query.slice(0, 38) + '…' : preset.query}</span>
          </button>
        ))}
        {query && (
          <button
            type="button"
            className="quick-tag"
            onClick={clearQuery}
            style={{ marginLeft: 'auto', color: 'var(--accent-rose)', borderColor: 'rgba(244,63,94,0.3)' }}
          >
            ✕ Reset
          </button>
        )}
      </div>
    </div>
  );
};
