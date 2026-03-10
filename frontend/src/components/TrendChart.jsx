function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

function TrendChart({ points = [], unit = 'ms' }) {
  if (!Array.isArray(points) || points.length === 0) {
    return <p className="chart-empty">No trend data yet.</p>
  }

  const width = 520
  const height = 220
  const paddingX = 34
  const paddingY = 24

  const values = points.map((item) => Number(item.value ?? 0))
  const labels = points.map((item, index) => item.label ?? `P${index + 1}`)

  const minVal = Math.min(...values)
  const maxVal = Math.max(...values)
  const spread = maxVal - minVal || Math.max(maxVal * 0.12, 1)
  const yMin = minVal - spread * 0.1
  const yMax = maxVal + spread * 0.1

  const xStep = points.length > 1 ? (width - paddingX * 2) / (points.length - 1) : 0

  const mapped = values.map((val, idx) => {
    const x = paddingX + idx * xStep
    const ratio = (val - yMin) / (yMax - yMin)
    const y = height - paddingY - ratio * (height - paddingY * 2)
    return { x, y, value: val, label: labels[idx] }
  })

  const linePath = mapped
    .map((pt, idx) => `${idx === 0 ? 'M' : 'L'} ${pt.x.toFixed(2)} ${pt.y.toFixed(2)}`)
    .join(' ')

  const areaPath = `${linePath} L ${mapped[mapped.length - 1].x.toFixed(2)} ${(height - paddingY).toFixed(
    2,
  )} L ${mapped[0].x.toFixed(2)} ${(height - paddingY).toFixed(2)} Z`

  const ticks = [0, 0.5, 1].map((ratio) => {
    const value = yMin + (yMax - yMin) * (1 - ratio)
    const y = paddingY + (height - paddingY * 2) * ratio
    return {
      y,
      value: clamp(value, -1e12, 1e12),
    }
  })

  return (
    <div className="trend-chart-wrap">
      <svg
        className="trend-chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Solve time trend chart"
        preserveAspectRatio="none"
      >
        {ticks.map((tick) => (
          <g key={tick.y}>
            <line className="chart-grid-line" x1={paddingX} x2={width - paddingX} y1={tick.y} y2={tick.y} />
            <text className="chart-axis-text" x={6} y={tick.y + 4}>
              {tick.value.toFixed(0)}
              {unit}
            </text>
          </g>
        ))}

        <path className="chart-area" d={areaPath} />
        <path className="chart-line" d={linePath} />

        {mapped.map((point) => (
          <g key={point.label}>
            <circle className="chart-point" cx={point.x} cy={point.y} r={4.2} />
            <text className="chart-x-text" x={point.x} y={height - 6} textAnchor="middle">
              {point.label}
            </text>
          </g>
        ))}
      </svg>
    </div>
  )
}

export default TrendChart
