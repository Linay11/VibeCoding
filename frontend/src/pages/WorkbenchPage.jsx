import { useEffect, useMemo, useState } from 'react'
import MetricCard from '../components/MetricCard.jsx'
import TrendChart from '../components/TrendChart.jsx'
import ComparisonChart from '../components/ComparisonChart.jsx'
import { getLatestRun, getScenarios, runExperiment } from '../services/optimizerApi.js'

function formatPct(value) {
  return `${(value * 100).toFixed(2)}%`
}

function formatMs(value) {
  return `${Math.round(value)} ms`
}

function buildDerivedTrend(runData) {
  const base = Number(runData?.metrics?.solveTimeMs ?? 0)
  if (!base) {
    return []
  }
  const multipliers = [1.18, 1.11, 1.05, 1.0, 0.94, 0.9]
  return multipliers.map((m, idx) => ({
    label: `R-${multipliers.length - idx}`,
    value: Math.max(0.1, base * m),
  }))
}

function buildComparisonRows(runData) {
  if (Array.isArray(runData?.comparison) && runData.comparison.length > 0) {
    return runData.comparison
      .map((item, index) => ({
        label: item.label ?? `Item ${index + 1}`,
        value: Number(item.value ?? 0),
      }))
      .filter((item) => Number.isFinite(item.value))
  }

  if (Array.isArray(runData?.strategies) && runData.strategies.length > 0) {
    return [...runData.strategies]
      .sort((a, b) => (a.rank ?? 0) - (b.rank ?? 0))
      .slice(0, 4)
      .map((item) => ({
        label: item.name,
        value: Number(item.cost ?? 0),
      }))
      .filter((item) => Number.isFinite(item.value))
  }

  return []
}

function WorkbenchPage() {
  const [scenarios, setScenarios] = useState([])
  const [selectedScenario, setSelectedScenario] = useState('')
  const [runData, setRunData] = useState(null)
  const [source, setSource] = useState('loading')
  const [notice, setNotice] = useState('')
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false

    async function loadScenarios() {
      setLoading(true)
      setError('')
      const response = await getScenarios()
      if (cancelled) {
        return
      }
      setScenarios(response.data)
      setSource(response.source)
      setNotice(response.notice ?? '')
      const defaultScenario = response.data[0]?.id ?? ''
      setSelectedScenario(defaultScenario)
      setLoading(false)
    }

    loadScenarios()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    async function loadLatestScenarioRun() {
      if (!selectedScenario) {
        return
      }
      setError('')
      const response = await getLatestRun(selectedScenario)
      if (cancelled) {
        return
      }
      setRunData(response.data)
      setSource(response.source)
      setNotice(response.notice ?? '')
    }

    loadLatestScenarioRun()
    return () => {
      cancelled = true
    }
  }, [selectedScenario])

  async function handleRefreshLatest() {
    if (!selectedScenario || running || loading) {
      return
    }
    setRefreshing(true)
    setError('')
    const response = await getLatestRun(selectedScenario)
    setRunData(response.data)
    setSource(response.source)
    setNotice(response.notice ?? '')
    setRefreshing(false)
  }

  async function handleRunNow() {
    if (!selectedScenario) {
      setError('No scenario selected. Please choose one before starting a run.')
      return
    }
    setRunning(true)
    setError('')
    const response = await runExperiment({ scenarioId: selectedScenario })
    if (!response.data) {
      setError('Run failed. Please verify backend endpoint readiness.')
      setRunning(false)
      return
    }
    setRunData(response.data)
    setSource(response.source)
    setNotice(response.notice ?? '')
    setRunning(false)
  }

  const selectedScenarioDetail = useMemo(
    () => scenarios.find((item) => item.id === selectedScenario),
    [scenarios, selectedScenario],
  )

  const trendPoints = useMemo(() => {
    if (!runData) {
      return []
    }
    if (Array.isArray(runData.trend) && runData.trend.length > 0) {
      return runData.trend
        .map((item, index) => ({
          label: item.label ?? `R${index + 1}`,
          value: Number(item.value ?? 0),
        }))
        .filter((item) => Number.isFinite(item.value))
    }
    return buildDerivedTrend(runData)
  }, [runData])

  const comparisonRows = useMemo(() => buildComparisonRows(runData), [runData])

  return (
    <div className="fade-in">
      <section className="workbench-head card">
        <div>
          <p className="eyebrow">Main Feature</p>
          <h1>Optimization Workbench</h1>
          <p className="lead">
            Trigger an experiment run, inspect metrics, and compare strategy outcomes.
          </p>
        </div>
        <p className="status-chip" role="status" aria-live="polite">
          Data source: <strong>{source === 'api' ? 'Backend API' : source === 'mock' ? 'Mock fallback' : 'Loading...'}</strong>
        </p>
      </section>

      {notice ? (
        <section className="notice-banner card" role="status" aria-live="polite">
          <p>{notice}</p>
        </section>
      ) : null}

      <section className="workbench-grid">
        <article className="card control-panel">
          <h2>Run Control</h2>
          <label className="field-label" htmlFor="scenario-picker">
            Scenario
          </label>
          <select
            id="scenario-picker"
            className="field"
            value={selectedScenario}
            onChange={(event) => setSelectedScenario(event.target.value)}
            disabled={loading || running}
          >
            {scenarios.length === 0 ? <option value="">No scenario available</option> : null}
            {scenarios.map((scenario) => (
              <option key={scenario.id} value={scenario.id}>
                {scenario.name}
              </option>
            ))}
          </select>

          <p className="field-help">
            {selectedScenarioDetail?.description ??
              (loading
                ? 'Loading scenarios...'
                : 'No scenario loaded. Check backend API or continue with demo data.')}
          </p>

          <div className="actions-row">
            <button
              className="btn btn-secondary"
              onClick={handleRefreshLatest}
              disabled={running || loading || refreshing || !selectedScenario}
            >
              {refreshing ? 'Refreshing...' : 'Refresh Latest'}
            </button>

            <button className="btn btn-primary" onClick={handleRunNow} disabled={running || loading || !selectedScenario}>
              {running ? 'Running...' : 'Run Experiment'}
            </button>
          </div>

          {source === 'mock' ? (
            <p className="field-help">
              You are viewing demo output. Connect backend endpoints to run real optimization jobs.
            </p>
          ) : null}

          {error ? (
            <p className="inline-error" role="alert">
              {error}
            </p>
          ) : null}
        </article>

        <section className="metrics-grid" aria-label="Run metrics">
          <MetricCard
            label="Solve Time"
            value={runData ? formatMs(runData.metrics.solveTimeMs) : '--'}
            hint="Average solve time from the latest run."
          />
          <MetricCard
            label="Infeasibility"
            value={runData ? formatPct(runData.metrics.infeasibilityRate) : '--'}
            hint="Lower values indicate better feasibility."
          />
          <MetricCard
            label="Suboptimality"
            value={runData ? formatPct(runData.metrics.suboptimality) : '--'}
            hint="Gap relative to reference solver objective."
          />
        </section>
      </section>

      <section className="card table-wrap">
        <div className="table-head">
          <h2>Strategy Snapshot</h2>
          <p>{runData ? `Run ID: ${runData.runId}` : 'No run yet. Start by clicking "Run Experiment".'}</p>
        </div>

        {runData ? (
          <div className="table-scroll">
            <table>
              <caption className="sr-only">Strategy output table</caption>
              <thead>
                <tr>
                  <th scope="col">Strategy</th>
                  <th scope="col">Feasible</th>
                  <th scope="col">Cost</th>
                  <th scope="col">Rank</th>
                </tr>
              </thead>
              <tbody>
                {runData.strategies.map((row) => (
                  <tr key={row.id}>
                    <td>{row.name}</td>
                    <td>{row.feasible ? 'Yes' : 'No'}</td>
                    <td>{row.cost.toFixed(3)}</td>
                    <td>{row.rank}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="empty-copy">
            No strategy table to display yet. Select a scenario, then run or refresh to load data.
          </p>
        )}
      </section>

      <section className="card chart-panel">
        <div className="table-head">
          <h2>Trend and Comparison</h2>
          <p>Chart data follows API payload when available; otherwise derived from current run data.</p>
        </div>

        <div className="chart-grid">
          <article className="chart-card">
            <h3 className="chart-title">Solve Time Trend</h3>
            <p className="chart-copy">Recent run trend (ms), useful for quick performance direction checks.</p>
            <TrendChart points={trendPoints} unit="ms" />
          </article>

          <article className="chart-card">
            <h3 className="chart-title">Result Comparison</h3>
            <p className="chart-copy">Compare strategy costs from the latest run.</p>
            <ComparisonChart rows={comparisonRows} formatValue={(value) => value.toFixed(3)} />
          </article>
        </div>
      </section>
    </div>
  )
}

export default WorkbenchPage
