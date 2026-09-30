import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { SearchBar } from './features/search/SearchBar';
import { SearchResultCard } from './features/search/SearchResultCard';
import { ComparisonViewer } from './features/comparison/ComparisonViewer';
import { TemporalProgressionViewer } from './features/comparison/TemporalProgressionViewer';
import { AnalystSummaryCard } from './features/comparison/AnalystSummaryCard';
import { SystemStatusView } from './features/system/SystemStatusView';
import {
  SearchResultItem,
  ComparisonResponse,
  ChangeDetectionResponse,
} from './types/api.types';
import {
  LOCATIONS,
  getSearchResults,
  matchQueryToPreset,
  buildComparison,
  buildChangeDetection,
  MOCK_COMPARISON_PARIS,
  MOCK_CHANGE_DETECTION_PARIS,
} from './mock/mockData';

// Simulated async delay for realism
const delay = (ms: number) => new Promise(res => setTimeout(res, ms));

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
  const [showTemporalProgression, setShowTemporalProgression] = useState(false);

  // Initial load — show all locations
  useEffect(() => {
    handleSearch('', 'q_all');
  }, []);

  const handleSearch = async (queryText: string, presetId: string) => {
    setIsSearching(true);
    setActiveQuery(queryText);
    await delay(600); // simulate network
    try {
      let pid = presetId;
      if (!pid) {
        pid = matchQueryToPreset(queryText).id;
      }
      const res = getSearchResults(pid);
      setResults(res);
    } finally {
      setIsSearching(false);
    }
  };

  const handleInspectTile = async (item: SearchResultItem) => {
    setActiveTab('comparison');
    setIsComparing(true);
    await delay(800); // simulate detection pipeline
    try {
      const loc = LOCATIONS.find(l => l.id === item.tile_ref.location_id);
      if (loc) {
        setComparison(buildComparison(loc));
        setChangeDetection(buildChangeDetection(loc));
      }
    } finally {
      setIsComparing(false);
    }
  };

  const handleRefreshDetection = async () => {
    setIsComparing(true);
    await delay(500);
    try {
      const loc = LOCATIONS.find(l => l.id === comparison.before.tile_ref.location_id);
      if (loc) {
        setChangeDetection(buildChangeDetection(loc));
      }
    } finally {
      setIsComparing(false);
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
                No satellite scenes match the query. Try a different search.
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
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '0.75rem' }}>
              <button className="secondary-btn" onClick={() => setShowTemporalProgression(value => !value)}>
                {showTemporalProgression ? 'Return to image comparison' : 'View temporal progression'}
              </button>
            </div>
            {showTemporalProgression ? <TemporalProgressionViewer locationId="kaziranga" /> : <ComparisonViewer
              comparison={comparison}
              changeDetection={changeDetection}
              onRefreshDetection={handleRefreshDetection}
              isLoading={isComparing}
            />}

            {!showTemporalProgression && changeDetection && (
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
