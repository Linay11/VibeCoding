function RunSummaryPanel({ scenario, sourceLabel, runModeLabel, modeReason, generatedTime }) {
  const reasonText = modeReason && modeReason.trim() ? modeReason : 'No additional reason provided.'
  const generatedText = generatedTime && generatedTime.trim() ? generatedTime : 'Not generated yet'

  return (
    <section className="card run-summary" aria-label="Run summary">
      <div className="summary-head">
        <h2>Result Summary</h2>
        <p>Key context for this result snapshot.</p>
      </div>

      <div className="summary-grid">
        <article className="summary-item">
          <p className="summary-label">Scenario</p>
          <p className="summary-value">{scenario || 'No scenario selected'}</p>
        </article>

        <article className="summary-item">
          <p className="summary-label">Data source</p>
          <p className="summary-value">
            <span className={`summary-badge source-${sourceLabel.toLowerCase().includes('fallback') ? 'fallback' : sourceLabel.toLowerCase().includes('api') ? 'api' : 'loading'}`}>
              {sourceLabel}
            </span>
          </p>
        </article>

        <article className="summary-item">
          <p className="summary-label">Run mode</p>
          <p className="summary-value">
            <span
              className={`summary-badge ${
                runModeLabel.toLowerCase().includes('real')
                  ? 'mode-real'
                  : runModeLabel.toLowerCase().includes('compat')
                    ? 'mode-compat'
                    : runModeLabel.toLowerCase().includes('fallback')
                      ? 'mode-fallback'
                      : 'mode-none'
              }`}
            >
              {runModeLabel}
            </span>
          </p>
        </article>

        <article className="summary-item">
          <p className="summary-label">Generated time</p>
          <p className="summary-value">{generatedText}</p>
        </article>

        <article className="summary-item summary-item-wide">
          <p className="summary-label">Mode reason</p>
          <p className="summary-value summary-reason">{reasonText}</p>
        </article>
      </div>
    </section>
  )
}

export default RunSummaryPanel
