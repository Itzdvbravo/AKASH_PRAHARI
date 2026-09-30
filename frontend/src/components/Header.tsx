interface HeaderProps {
  activeTab: 'search' | 'comparison' | 'system';
  onSelectTab: (tab: 'search' | 'comparison' | 'system') => void;
  mockMode?: boolean;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, onSelectTab, mockMode = true }) => {
  return (
    <header className="top-header">
      <div className="brand-section">
        <div className="brand-logo-icon">AP</div>
        <div className="brand-title">
          TerraEyes
          <span className="brand-subtitle">Akash Prahari Geospatial</span>
        </div>
      </div>

      <nav className="nav-tabs" aria-label="Main Navigation">
        <button
          className={`nav-tab-btn ${activeTab === 'search' ? 'active' : ''}`}
          onClick={() => onSelectTab('search')}
          type="button"
        >
          <span>🔍</span>
          <span>Search & Retrieval</span>
        </button>
        <button
          className={`nav-tab-btn ${activeTab === 'comparison' ? 'active' : ''}`}
          onClick={() => onSelectTab('comparison')}
          type="button"
        >
          <span>🛰️</span>
          <span>Temporal Comparison</span>
        </button>
        <button
          className={`nav-tab-btn ${activeTab === 'system' ? 'active' : ''}`}
          onClick={() => onSelectTab('system')}
          type="button"
        >
          <span>⚡</span>
          <span>System Health</span>
        </button>
      </nav>

      <div className="header-meta">
        <div className="status-indicator">
          <span className="status-dot"></span>
          <span>{mockMode ? 'Mock Data Active' : 'Connected to Core'}</span>
        </div>
      </div>
    </header>
  );
};
