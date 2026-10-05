import { useEffect, useState } from 'react';
import { HealthResponse } from '../../types/api.types';
import { fetchHealth } from '../../api/system.api';

export const SystemStatusView: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetchHealth()
      .then((data) => setHealth(data))
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>Querying Subsystems...</div>;
  }

  if (error || !health) {
    return <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--accent-rose)' }}>Unable to query system health</div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h2 className="section-title">Subsystem Diagnostic Status</h2>
        <span className="badge badge-emerald">Operational</span>
      </div>

      <div className="system-status-grid">
        <div className="system-status-card">
          <div className="system-status-header">
            <span className="system-status-title">Embedding Subsystem</span>
            <span className="badge badge-cyan">Active</span>
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
            {health.embedding_model}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
            Zero-shot semantic vector generation for multispectral satellite tiles
          </div>
        </div>

        <div className="system-status-card">
          <div className="system-status-header">
            <span className="system-status-title">Change Detection Engine</span>
            <span className="badge badge-emerald">Online</span>
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
            {health.change_detector}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
            Bi-temporal coregistration & pixel-level differential analysis
          </div>
        </div>

        <div className="system-status-card">
          <div className="system-status-header">
            <span className="system-status-title">Vector Index (FAISS)</span>
            <span className="badge badge-cyan">{health.faiss_index_size.toLocaleString()} vectors</span>
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Cosine similarity index for fast top-k tile retrieval
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
            Indexed tiles: {health.db_tiles.toLocaleString()} scenes
          </div>
        </div>

        <div className="system-status-card">
          <div className="system-status-header">
            <span className="system-status-title">Dataset Source</span>
            <span className="badge badge-emerald">Ready</span>
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            {health.embedding_model === 'mock' ? 'Synthetic prototype catalog' : 'DynamicEarthNet PlanetFusion archive'}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
            {health.embedding_model === 'mock'
              ? 'Demo scenes and dates for prototype mode'
              : 'Monthly imagery and seven-class land-cover reference annotations'}
          </div>
        </div>
      </div>
    </div>
  );
};
