function ComparisonChart({ rows = [], formatValue }) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return <p className="chart-empty">No comparison data yet.</p>
  }

  const max = Math.max(...rows.map((item) => Number(item.value ?? 0)), 1e-9)
  const formatter =
    formatValue ??
    ((value) => {
      return Number(value).toFixed(3)
    })

  return (
    <div className="comparison-chart">
      {rows.map((row) => {
        const value = Number(row.value ?? 0)
        const pct = Math.max(8, (value / max) * 100)
        return (
          <div className="comparison-row" key={row.label}>
            <div className="comparison-head">
              <span>{row.label}</span>
              <strong>{formatter(value)}</strong>
            </div>
            <div className="comparison-track" aria-hidden="true">
              <div className="comparison-fill" style={{ width: `${pct}%` }} />
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default ComparisonChart
