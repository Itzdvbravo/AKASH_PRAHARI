import { AnalystSummary } from '../../types/api.types';

interface AnalystSummaryCardProps {
  summary: AnalystSummary;
  processingMs?: number;
}

export const AnalystSummaryCard: React.FC<AnalystSummaryCardProps> = ({ summary, processingMs = 184 }) => {
  const confidencePct = (summary.confidence.score * 100).toFixed(1);
  const changedPct = (summary.changed_pixel_fraction * 100).toFixed(2);

  return (
    <div className="ui-card summary-panel" style={{ marginTop: '1.25rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h3 className="section-title">Analyst Change Detection Summary</h3>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Processed in {processingMs}ms
        </span>
      </div>

      <div className="metrics-summary-grid">
        <div className="metric-item">
          <span className="metric-title">Detected Change</span>
          <span className="metric-value" style={{ fontSize: '1rem', color: 'var(--accent-rose)' }}>
            {summary.change_type}
          </span>
          <span className="metric-sub">{summary.location}</span>
        </div>

        <div className="metric-item">
          <span className="metric-title">Model Confidence</span>
          <span className="metric-value" style={{ color: 'var(--accent-emerald)' }}>
            {confidencePct}%
          </span>
          <span className="metric-sub">Method: {summary.confidence.method}</span>
        </div>

        <div className="metric-item">
          <span className="metric-title">Area Affected</span>
          <span className="metric-value">
            {changedPct}%
          </span>
          <span className="metric-sub">Pixel delta fraction</span>
        </div>

        <div className="metric-item">
          <span className="metric-title">Earliest Detection</span>
          <span className="metric-value" style={{ color: 'var(--accent-cyan)' }}>
            {summary.earliest_detectable_change ?? 'N/A'}
          </span>
          <span className="metric-sub">From temporal archive</span>
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.75rem' }}>
        <span><strong>Detector:</strong> {summary.detector}</span>
        <span><strong>Source:</strong> {summary.source_provenance}</span>
      </div>
    </div>
  );
};
