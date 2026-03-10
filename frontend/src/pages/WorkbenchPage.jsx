import { useEffect, useMemo, useState } from 'react'
import MetricCard from '../components/MetricCard.jsx'
import TrendChart from '../components/TrendChart.jsx'
import ComparisonChart from '../components/ComparisonChart.jsx'
import RunSummaryPanel from '../components/RunSummaryPanel.jsx'
import DashboardStatePanel from '../components/DashboardStatePanel.jsx'
import { getLatestRun, getScenarios, runExperiment } from '../services/optimizerApi.js'

function formatPct(value) {
  return `${(value * 100).toFixed(2)}%`
}

function formatMs(value) {
  return `${Math.round(value)} ms`
}

function formatGeneratedTime(value) {
  if (!value) {
    return 'Not generated yet'
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return String(value)
  }
  return date.toLocaleString()
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

function getSourceLabel(source) {
  if (source === 'api') {
    return 'Backend API'
  }
  if (source === 'fallback') {
    return 'Frontend fallback'
  }
  return 'Loading'
}

function getModeLabel(mode) {
  if (mode === 'real') {
    return 'Real run'
  }
  if (mode === 'compat') {
    return 'Compat run'
  }
  if (mode === 'fallback') {
    return 'Fallback demo'
  }
  return 'Not available'
}

function resolveDashboardState({ loading, refreshing, running, error, runData, source, runMode, modeReason }) {
  if (error) {
    return {
      key: 'error',
      title: 'Action error',
      description: error,
    }
  }

  if (loading && !runData) {
    return {
      key: 'loading',
      title: 'Loading baseline data',
      description: 'Fetching scenarios and latest run data from backend adapter.',
    }
  }

  if (running) {
    return {
      key: 'loading',
      title: 'Run in progress',
      description: 'Run Experiment is executing. Results will refresh when the response returns.',
    }
  }

  if (refreshing) {
    return {
      key: 'loading',
      title: 'Refreshing latest run',
      description: 'Reading latest backend run for the selected scenario.',
    }
  }

  if (!runData) {
    return {
      key: 'empty',
      title: 'No run data yet',
      description: 'Select a scenario and run an experiment, or refresh latest to load existing data.',
    }
  }

  if (source === 'fallback' || runMode === 'fallback') {
    return {
      key: 'fallback',
      title: 'Frontend fallback mode',
      description: modeReason || 'The UI is showing local fallback data because backend data is unavailable for this step.',
    }
  }

  if (source === 'api' && runMode === 'compat') {
    return {
      key: 'compat',
      title: 'Backend compatibility mode',
      description: modeReason || 'Response came from backend adapter compatibility mode, not full real solver execution.',
    }
  }

  if (source === 'api' && runMode === 'real') {
    return {
      key: 'real',
      title: 'Backend real execution',
      description: 'Current result comes from a real backend optimization run.',
    }
  }

  return {
    key: 'empty',
    title: 'State pending',
    description: 'Waiting for a complete result context.',
  }
}

function WorkbenchPage() {
  const [scenarios, setScenarios] = useState([])
  const [selectedScenario, setSelectedScenario] = useState('')
  const [runData, setRunData] = useState(null)
  const [source, setSource] = useState('loading')
  const [runMode, setRunMode] = useState('none')
  const [runModeReason, setRunModeReason] = useState('')
  const [notice, setNotice] = useState('')
  const [noticeTone, setNoticeTone] = useState('info')
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')

  function applyRunResponse(response) {
    if (!response) {
      return
    }
    setRunData(response.data ?? null)
    setSource(response.source ?? 'loading')
    setRunMode(response.mode ?? response.data?.adapterMode ?? 'none')
    setRunModeReason(response.modeReason ?? response.data?.adapterNote ?? '')
    setNotice(response.notice ?? '')
    setNoticeTone(response.noticeTone ?? 'info')
  }

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
      setNoticeTone(response.noticeTone ?? 'info')
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
      applyRunResponse(response)
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
    try {
      const response = await getLatestRun(selectedScenario)
      applyRunResponse(response)
    } finally {
      setRefreshing(false)
    }
  }

  async function handleRunNow() {
    if (!selectedScenario) {
      setError('No scenario selected. Please choose one before starting a run.')
      return
    }
    setRunning(true)
    setError('')
    try {
      const response = await runExperiment({ scenarioId: selectedScenario })
      if (!response.data) {
        setError('Run failed: backend did not return usable data.')
        return
      }
      applyRunResponse(response)
    } finally {
      setRunning(false)
    }
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

  const summaryScenario = selectedScenarioDetail?.name ?? selectedScenario ?? 'Not selected'
  const summarySource = getSourceLabel(source)
  const summaryMode = getModeLabel(runMode)
  const summaryReason =
    runModeReason ||
    (runMode === 'real'
      ? 'Real solver execution completed via backend adapter.'
      : runMode === 'compat'
        ? 'Backend adapter returned compatibility run output.'
        : runMode === 'fallback'
          ? 'Frontend generated fallback run due to API unavailability.'
          : 'No run reason available yet.')
  const summaryGenerated = formatGeneratedTime(runData?.generatedAt)

  const dashboardState = resolveDashboardState({
    loading,
    refreshing,
    running,
    error,
    runData,
    source,
    runMode,
    modeReason: summaryReason,
  })

  return (
    <div className="fade-in">
      <section className="workbench-head card">
        <div>
          <p className="eyebrow">Optimization Experiment Dashboard</p>
          <h1>Workbench</h1>
          <p className="lead">
            Run experiments, inspect solver behavior, and explain results with clear runtime context.
          </p>
        </div>
      </section>

      <RunSummaryPanel
        scenario={summaryScenario}
        sourceLabel={summarySource}
        runModeLabel={summaryMode}
        modeReason={summaryReason}
        generatedTime={summaryGenerated}
      />

      <DashboardStatePanel stateKey={dashboardState.key} title={dashboardState.title} description={dashboardState.description} />

      {notice ? (
        <section className={`notice-banner card notice-${noticeTone}`} role="status" aria-live="polite">
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
                : 'No scenario loaded. Check backend API or continue with fallback demo data.')}
          </p>

          <div className="actions-row">
            <button
              className="btn btn-secondary"
              onClick={handleRefreshLatest}
              disabled={running || loading || refreshing || !selectedScenario}
            >
              {refreshing ? 'Refreshing...' : 'Refresh Latest'}
            </button>

            <button className="btn btn-primary" onClick={handleRunNow} disabled={running || refreshing || loading || !selectedScenario}>
              {running ? 'Running...' : 'Run Experiment'}
            </button>
          </div>

          {source === 'fallback' ? (
            <p className="field-help">
              Fallback mode is active. Data is interactive, but it is not a confirmed backend latest run.
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
            hint="How fast this run finished."
            tooltip="Elapsed solver time for this run. Lower is generally better for responsiveness."
          />
          <MetricCard
            label="Infeasibility Rate"
            value={runData ? formatPct(runData.metrics.infeasibilityRate) : '--'}
            hint="How often constraints are broken."
            tooltip="Share of outcomes that violate constraints. Lower means more reliable solutions."
          />
          <MetricCard
            label="Suboptimality"
            value={runData ? formatPct(runData.metrics.suboptimality) : '--'}
            hint="How far from best-known objective."
            tooltip="Gap between current objective and reference best objective. Lower indicates better quality."
          />
        </section>
      </section>

      <section className="card analysis-intro">
        <h2>Analysis Flow</h2>
        <p>
          Step 1 checks runtime trend, Step 2 compares strategy cost, and Step 3 reviews full strategy rows for explanation details.
        </p>
      </section>

      <section className="analysis-grid">
        <section className="card chart-panel">
          <div className="table-head">
            <h2>Step 1-2: Trend and Comparison</h2>
            <p>Charts prefer backend payload values and only derive locally as a fallback safeguard.</p>
          </div>

          <div className="chart-grid">
            <article className="chart-card">
              <h3 className="chart-title">1) Solve Time Trend</h3>
              <p className="chart-copy">Recent solve-time direction (ms) for quick performance trend checks.</p>
              <TrendChart points={trendPoints} unit="ms" />
            </article>

            <article className="chart-card">
              <h3 className="chart-title">2) Cost Comparison</h3>
              <p className="chart-copy">Relative cost across strategies in the latest run snapshot.</p>
              <ComparisonChart rows={comparisonRows} formatValue={(value) => value.toFixed(3)} />
            </article>
          </div>
        </section>

        <section className="card table-wrap strategy-panel">
          <div className="table-head">
            <h2>Step 3: Strategy Table</h2>
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
            <p className="empty-copy">No strategy table to display yet. Select a scenario, then run or refresh to load data.</p>
          )}
        </section>
      </section>
    </div>
  )
}

export default WorkbenchPage
