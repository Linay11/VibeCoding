function MetricCard({ label, value, hint, tooltip }) {
  return (
    <article className="metric-card">
      <p className="metric-label-row">
        <span className="metric-label">{label}</span>
        {tooltip ? (
          <span className="metric-tip" title={tooltip} aria-label={`${label} explanation`}>
            i
          </span>
        ) : null}
      </p>
      <p className="metric-value">{value}</p>
      <p className="metric-hint">{hint}</p>
    </article>
  )
}

export default MetricCard
