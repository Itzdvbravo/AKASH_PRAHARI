import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { SearchBar } from './features/search/SearchBar';
import { SearchResultCard } from './features/search/SearchResultCard';
import { ComparisonViewer } from './features/comparison/ComparisonViewer';
import { AnalystSummaryCard } from './features/comparison/AnalystSummaryCard';
import { SystemStatusView } from './features/system/SystemStatusView';
import { searchTiles } from './api/search.api';
import { fetchComparison, detectChange } from './api/comparison.api';
import {
  SearchResultItem,
  Sensor,
  ComparisonResponse,
  ChangeDetectionResponse
} from './types/api.types';
import {
  MOCK_COMPARISON_PARIS,
  MOCK_CHANGE_DETECTION_PARIS,
  LOCATIONS,
  buildComparison,
  buildChangeDetection
} from './mock/mockData';

export default function App() {
  const [activeTab, setActiveTab] = useState<'search' | 'comparison' | 'system'>('search');
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [activeQuery, setActiveQuery] = useState('');

  // Comparison State
  const [comparison, setComparison] = useState<ComparisonResponse>(MOCK_COMPARISON_PARIS);
  const [changeDetection, setChangeDetection] = useState<ChangeDetectionResponse | null>(
    MOCK_CHANGE_DETECTION_PARIS
  );
  const [isComparing, setIsComparing] = useState(false);

  // Initial search load
  useEffect(() => {
    handleSearch('', 'any', 'all');
  }, []);

  const handleSearch = async (queryText: string, sensor: Sensor, location: string) => {
    setIsSearching(true);
    setActiveQuery(queryText);
    try {
      const res = await searchTiles({
        query_type: 'text',
        query_text: queryText,
        filters: {
          sensor,
          location
        }
      });
      setResults(res.results);
    } catch {
      // Handled inside searchTiles with fallback
    } finally {
      setIsSearching(false);
    }
  };

  const handleInspectTile = async (item: SearchResultItem) => {
    setActiveTab('comparison');
    setIsComparing(true);
    try {
      const comp = await fetchComparison({
        location_id: item.tile_ref.location_id,
        tile_id: item.tile_ref.tile_id,
        date_before: item.available_dates[0] || '2018-03-10',
        date_after: item.tile_ref.date,
        sensor: item.tile_ref.sensor
      });
      setComparison(comp);

      const det = await detectChange({
        location_id: item.tile_ref.location_id,
        tile_id: item.tile_ref.tile_id
      });
      setChangeDetection(det);
    } finally {
      setIsComparing(false);
    }
  };

  const handleRefreshDetection = async () => {
    setIsComparing(true);
    try {
      const det = await detectChange({
        location_id: comparison.before.tile_ref.location_id,
        tile_id: comparison.before.tile_ref.tile_id
      });
      setChangeDetection(det);
    } finally {
      setIsComparing(false);
    }
  };

  const handleLocationChange = (locId: string) => {
    const loc = LOCATIONS.find(l => l.id === locId);
    if (loc) {
      setComparison(buildComparison(loc));
      setChangeDetection(buildChangeDetection(loc));
    }
  };

  const resultLabel = activeQuery
    ? `"${activeQuery.length > 60 ? activeQuery.slice(0, 60) + '…' : activeQuery}"`
    : 'All Locations';

  return (
    <div className="app-container">
      <Header activeTab={activeTab} onSelectTab={setActiveTab} mockMode={true} />

      <main className="main-content">
        {activeTab === 'search' && (
          <section aria-label="Search and Results">
            <SearchBar onSearch={handleSearch} isLoading={isSearching} />

            <div className="results-header">
              <div>
                <h2 className="section-title">Retrieved Satellite Scenes</h2>
                {activeQuery && (
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                    Query: {resultLabel}
                  </p>
                )}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                {isSearching && (
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span className="spinner" /> Retrieving…
                  </span>
                )}
                <span className="results-count">
                  {results.length} candidate tile{results.length === 1 ? '' : 's'} found
                </span>
              </div>
            </div>

            {results.length === 0 && !isSearching ? (
              <div className="ui-card" style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                <div style={{ fontSize: '2rem', marginBottom: '0.75rem' }}>🛰️</div>
                No satellite scenes match the selected filters or query. Try resetting filters.
              </div>
            ) : (
              <div className="results-grid">
                {results.map((item) => (
                  <SearchResultCard
                    key={`${item.tile_ref.tile_id}_${item.rank}`}
                    item={item}
                    onInspect={handleInspectTile}
                  />
                ))}
              </div>
            )}
          </section>
        )}

        {activeTab === 'comparison' && (
          <section aria-label="Temporal Comparison">
            <ComparisonViewer
              comparison={comparison}
              changeDetection={changeDetection}
              onRefreshDetection={handleRefreshDetection}
              onLocationChange={handleLocationChange}
              isLoading={isComparing}
            />

            {changeDetection && (
              <AnalystSummaryCard
                summary={changeDetection.summary}
                processingMs={changeDetection.processing_ms}
              />
            )}
          </section>
        )}

        {activeTab === 'system' && (
          <section aria-label="System Health">
            <SystemStatusView />
          </section>
        )}
      </main>
    </div>
  );
}
