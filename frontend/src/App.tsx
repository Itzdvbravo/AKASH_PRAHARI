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
  GeoBBox,
  ComparisonResponse,
  ChangeDetectionResponse,
} from './types/api.types';
import { searchTiles } from './api/search.api';
import { fetchComparison, detectChange } from './api/comparison.api';
import { fetchHealth } from './api/system.api';
import { LOCATIONS, QUERY_PRESETS, matchQueryToPreset } from './mock/mockData';

const SEARCH_RESULT_LIMIT = 100;
const RESULTS_PER_PAGE = 12;

export default function App() {
  const [activeTab, setActiveTab] = useState<'search' | 'comparison' | 'system'>('search');
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [activeQuery, setActiveQuery] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [embeddingModel, setEmbeddingModel] = useState<string | null>(null);
  const [indexedTileCount, setIndexedTileCount] = useState<number | null>(null);
  const [resultsPage, setResultsPage] = useState(1);

  // Comparison State
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [comparisonGeoBBox, setComparisonGeoBBox] = useState<GeoBBox | null>(null);
  const [changeDetection, setChangeDetection] = useState<ChangeDetectionResponse | null>(null);
  const [isComparing, setIsComparing] = useState(false);
  const [showTemporalProgression, setShowTemporalProgression] = useState(false);

  // Initial load — show all locations
  useEffect(() => {
    void handleSearch('', 'q_all');
    fetchHealth().then(health => {
      setEmbeddingModel(health.embedding_model);
      setIndexedTileCount(health.faiss_index_size);
    }).catch(() => undefined);
  }, []);

  const handleSearch = async (queryText: string, presetId: string) => {
    setIsSearching(true);
    setResultsPage(1);
    setActiveQuery(queryText);
    setError(null);
    try {
      const preset = QUERY_PRESETS.find(candidate => candidate.id === presetId)
        ?? matchQueryToPreset(queryText);
      const locationFilter = embeddingModel === 'mock' && preset.id !== 'q_all' && preset.locationIds.length < 20
        ? preset.locationIds[0]
        : undefined;
      const response = await searchTiles({
        query_type: 'text',
        query_text: queryText,
        filters: { sensor: 'any', location: locationFilter },
        top_k: SEARCH_RESULT_LIMIT,
      });
      setResults(response.results);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Search failed. Check that the API server is running.');
    } finally {
      setIsSearching(false);
    }
  };

  const handleInspectTile = async (item: SearchResultItem) => {
    setActiveTab('comparison');
    setComparisonGeoBBox(item.geo_bbox);
    setIsComparing(true);
    setError(null);
    try {
      const dates = item.available_dates;
      if (dates.length < 2) throw new Error('This result has fewer than two available dates to compare.');
      const pair = await fetchComparison({
        location_id: item.tile_ref.location_id,
        tile_id: item.tile_ref.tile_id,
        date_before: dates[0],
        date_after: dates[dates.length - 1],
        sensor: item.tile_ref.sensor,
      });
      setComparison(pair);
      setChangeDetection(await detectChange({ comparison_id: pair.comparison_id }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not load the comparison.');
    } finally {
      setIsComparing(false);
    }
  };

  const handleRefreshDetection = async () => {
    setIsComparing(true);
    setError(null);
    try {
      if (!comparison) return;
      setChangeDetection(await detectChange({ comparison_id: comparison.comparison_id }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Change detection failed.');
    } finally {
      setIsComparing(false);
    }
  };

  const resultLabel = activeQuery
    ? `"${activeQuery.length > 60 ? activeQuery.slice(0, 60) + '…' : activeQuery}"`
    : 'All Locations';
  const pageCount = Math.ceil(results.length / RESULTS_PER_PAGE);
  const pageResults = results.slice(
    (resultsPage - 1) * RESULTS_PER_PAGE,
    resultsPage * RESULTS_PER_PAGE,
  );
  const rangeStart = results.length === 0 ? 0 : (resultsPage - 1) * RESULTS_PER_PAGE + 1;
  const rangeEnd = Math.min(resultsPage * RESULTS_PER_PAGE, results.length);

  return (
    <div className="app-container">
      <Header activeTab={activeTab} onSelectTab={setActiveTab} mockMode={embeddingModel === 'mock'} />

      <main className="main-content">
        {activeTab === 'search' && (
          <section aria-label="Search and Results">
            <SearchBar onSearch={handleSearch} isLoading={isSearching} />
            {error && <div className="ui-card" role="alert" style={{ color: 'var(--accent-rose)', marginTop: '1rem' }}>{error}</div>}

            <div className="results-header">
              <div>
                <h2 className="section-title">Retrieved Satellite Scenes</h2>
                {embeddingModel === 'clip_vit_b32' && indexedTileCount !== null && (
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                    Searching {indexedTileCount} indexed tiles from the OSCD training split; held-out test cities are excluded.
                  </p>
                )}
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
                  {results.length} ranked candidate{results.length === 1 ? '' : 's'}
                  {indexedTileCount !== null && ` · ${indexedTileCount} indexed tiles`}
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
                {pageResults.map((item) => (
                  <SearchResultCard
                    key={`${item.tile_ref.tile_id}_${item.rank}`}
                    item={item}
                    onInspect={handleInspectTile}
                  />
                ))}
              </div>
            )}
            {pageCount > 1 && (
              <nav className="results-pagination" aria-label="Search result pages">
                <span className="results-count">
                  Showing {rangeStart}–{rangeEnd} of {results.length} ranked candidates
                </span>
                <div className="pagination-controls">
                  <button
                    type="button"
                    className="secondary-btn"
                    onClick={() => setResultsPage(page => Math.max(1, page - 1))}
                    disabled={resultsPage === 1}
                  >
                    Previous
                  </button>
                  <span className="results-count" aria-live="polite">Page {resultsPage} of {pageCount}</span>
                  <button
                    type="button"
                    className="secondary-btn"
                    onClick={() => setResultsPage(page => Math.min(pageCount, page + 1))}
                    disabled={resultsPage === pageCount}
                  >
                    Next
                  </button>
                </div>
              </nav>
            )}
          </section>
        )}

        {activeTab === 'comparison' && comparison && (
          <section aria-label="Temporal Analysis">
            {embeddingModel === 'mock' && (
              <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '0.75rem' }}>
                <button className="secondary-btn" onClick={() => setShowTemporalProgression(value => !value)}>
                  {showTemporalProgression ? 'Open image comparison' : 'View temporal progression'}
                </button>
              </div>
            )}
            {embeddingModel === 'mock' && showTemporalProgression && LOCATIONS.some(location => location.id === comparison.before.tile_ref.location_id) ? <TemporalProgressionViewer locationId={comparison.before.tile_ref.location_id} /> : <ComparisonViewer
              comparison={comparison}
              geoBBox={comparisonGeoBBox ?? undefined}
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

        {activeTab === 'comparison' && !comparison && (
          <section aria-label="Temporal Analysis" className="ui-card">Choose a search result and select “Inspect & Compare Changes” to run the live comparison.</section>
        )}

        {activeTab === 'comparison' && error && <div className="ui-card" role="alert" style={{ color: 'var(--accent-rose)', marginTop: '1rem' }}>{error}</div>}

        {activeTab === 'system' && (
          <section aria-label="System Health">
            <SystemStatusView />
          </section>
        )}
      </main>
    </div>
  );
}
