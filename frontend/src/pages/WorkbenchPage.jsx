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
    return '尚未生成'
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return String(value)
  }
  return date.toLocaleString()
}

function formatBool(value) {
  return value ? '是' : '否'
}

function formatComparisonCost(value) {
  return Number(value).toFixed(3)
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
        label: item.label ?? `项目 ${index + 1}`,
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
    return '后端 API'
  }
  if (source === 'fallback') {
    return '前端回退'
  }
  return '加载中'
}

function getModeLabel(mode) {
  if (mode === 'real') {
    return '真实执行'
  }
  if (mode === 'compat') {
    return '兼容模式'
  }
  if (mode === 'fallback') {
    return '回退演示'
  }
  return '不可用'
}

function resolveDashboardState({ loading, refreshing, running, error, runData, source, runMode, modeReason }) {
  if (error) {
    return {
      key: 'error',
      title: '操作异常',
      description: error,
    }
  }

  if (loading && !runData) {
    return {
      key: 'loading',
      title: '正在加载基础数据',
      description: '正在从后端适配器获取场景列表和最新运行结果。',
    }
  }

  if (running) {
    return {
      key: 'loading',
      title: '运行进行中',
      description: '实验已启动，返回结果后会自动刷新。',
    }
  }

  if (refreshing) {
    return {
      key: 'loading',
      title: '正在刷新最新结果',
      description: '正在读取当前场景的后端最新运行数据。',
    }
  }

  if (!runData) {
    return {
      key: 'empty',
      title: '暂无运行数据',
      description: '请选择场景并启动实验，或刷新最新结果加载已有数据。',
    }
  }

  if (source === 'fallback' || runMode === 'fallback') {
    return {
      key: 'fallback',
      title: '前端回退模式',
      description: modeReason || '当前步骤无法获取后端数据，界面展示本地回退结果。',
    }
  }

  if (source === 'api' && runMode === 'compat') {
    return {
      key: 'compat',
      title: '后端兼容模式',
      description: modeReason || '当前结果来自后端适配器兼容路径，不是完整真实求解。',
    }
  }

  if (source === 'api' && runMode === 'real') {
    return {
      key: 'real',
      title: '后端真实执行',
      description: '当前结果来自后端真实优化运行。',
    }
  }

  return {
    key: 'empty',
    title: '状态等待中',
    description: '正在等待完整结果上下文。',
  }
}

function WorkbenchPage() {
  const [scenarios, setScenarios] = useState([])
  const [selectedScenario, setSelectedScenario] = useState('')
  const [selectedRunMode, setSelectedRunMode] = useState('exact')
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
      setError('尚未选择场景，请先选择后再开始运行。')
      return
    }
    setRunning(true)
    setError('')
    try {
      const response = await runExperiment({
        scenarioId: selectedScenario,
        runMode: selectedScenario === 'power-118' ? selectedRunMode : 'exact',
      })
      if (!response.data) {
        setError('运行失败：后端未返回可用数据。')
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
  const isPower118Scenario = selectedScenario === 'power-118'

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

  const summaryScenario = selectedScenarioDetail?.name ?? selectedScenario ?? '未选择'
  const summarySource = getSourceLabel(source)
  const summaryMode = getModeLabel(runMode)
  const requestedModeLabel = runData?.requestedRunMode ?? (isPower118Scenario ? selectedRunMode : null)
  const actualModeLabel = runData?.solverModeUsed ?? null
  const modeSwitchLabel =
    requestedModeLabel && actualModeLabel && requestedModeLabel !== actualModeLabel
      ? `请求模式 ${requestedModeLabel}，实际使用 ${actualModeLabel}。`
      : ''
  const solverModeLabel = runData?.solverModeUsed ? `求解模式：${runData.solverModeUsed}。` : ''
  const diagnosticBits = [
    modeSwitchLabel,
    runData?.modelVersion ? `模型版本：${runData.modelVersion}。` : '',
    runData?.featureSchemaVersion ? `特征版本：${runData.featureSchemaVersion}。` : '',
    runData?.mlConfidence != null ? `ML 置信度：${(Number(runData.mlConfidence) * 100).toFixed(1)}%。` : '',
    runData?.repairApplied != null ? `修复已应用：${formatBool(runData.repairApplied)}。` : '',
    runData?.objectiveValue != null ? `目标值：${Number(runData.objectiveValue).toFixed(3)}。` : '',
    runData?.runtimeMs != null ? `运行耗时：${Math.round(Number(runData.runtimeMs))} ms。` : '',
    runData?.feasible != null ? `可行性：${formatBool(runData.feasible)}。` : '',
    runData?.fallbackReason ? `回退原因：${runData.fallbackReason}。` : '',
  ]
    .filter(Boolean)
    .join(' ')
  const baseSummaryReason =
    runModeReason ||
    (runMode === 'real'
      ? '后端适配器已完成真实求解执行。'
      : runMode === 'compat'
        ? '后端适配器返回兼容模式结果。'
        : runMode === 'fallback'
          ? '由于 API 不可用，前端生成了回退结果。'
          : '当前暂无可用运行说明。')
  const summaryReason = [solverModeLabel, diagnosticBits, baseSummaryReason].filter(Boolean).join(' ')
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
          <p className="eyebrow">优化实验控制台</p>
          <h1>实验台</h1>
          <p className="lead">
            在一个页面里完成实验运行、行为观察与结果解释，确保上下文清晰可追踪。
          </p>
        </div>
      </section>

      <section className="workbench-stage">
        <article className="card control-panel">
          <h2>运行控制</h2>
          <label className="field-label" htmlFor="scenario-picker">
            场景
          </label>
          <select
            id="scenario-picker"
            className="field"
            value={selectedScenario}
            onChange={(event) => setSelectedScenario(event.target.value)}
            disabled={loading || running}
          >
            {scenarios.length === 0 ? <option value="">暂无可用场景</option> : null}
            {scenarios.map((scenario) => (
              <option key={scenario.id} value={scenario.id}>
                {scenario.name}
              </option>
            ))}
          </select>

          <p className="field-help">
            {selectedScenarioDetail?.description ??
              (loading
                ? '正在加载场景...'
                : '场景未加载。请检查后端 API，或继续使用回退演示数据。')}
          </p>

          {isPower118Scenario ? (
            <>
              <label className="field-label" htmlFor="run-mode-picker">
                求解模式
              </label>
              <select
                id="run-mode-picker"
                className="field"
                value={selectedRunMode}
                onChange={(event) => setSelectedRunMode(event.target.value)}
                disabled={loading || running}
              >
                <option value="exact">精确</option>
                <option value="hybrid">混合</option>
                <option value="ml">仅 ML</option>
              </select>
              <p className="field-help">混合模式会优先使用 ML 热启动，并在远端求解器可用时接入精确 SCUC。</p>
            </>
          ) : null}

          <div className="actions-row">
            <button
              className="btn btn-secondary"
              onClick={handleRefreshLatest}
              disabled={running || loading || refreshing || !selectedScenario}
            >
              {refreshing ? '刷新中...' : '刷新最新结果'}
            </button>

            <button className="btn btn-primary" onClick={handleRunNow} disabled={running || refreshing || loading || !selectedScenario}>
              {running ? '运行中...' : '开始运行'}
            </button>
          </div>

          {source === 'fallback' ? (
            <p className="field-help">
              当前处于前端回退模式。你仍可交互查看，但该数据不是后端确认的最新运行结果。
            </p>
          ) : null}

          {error ? (
            <p className="inline-error" role="alert">
              {error}
            </p>
          ) : null}
        </article>

        <section className="workbench-main">
          <RunSummaryPanel
            scenario={summaryScenario}
            sourceLabel={summarySource}
            sourceCode={source}
            runModeLabel={summaryMode}
            modeCode={runMode}
            modeReason={summaryReason}
            generatedTime={summaryGenerated}
          />

          {isPower118Scenario && runData ? (
            <section className="card power118-diagnostics" aria-label="电力 118 诊断信息">
              <h3>电力 118 诊断</h3>
              <div className="power118-diagnostics-grid">
                <p>{`请求模式：${requestedModeLabel ?? '未知'}`}</p>
                <p>{`实际模式：${actualModeLabel ?? '未知'}`}</p>
                <p>{`可行性：${formatBool(Boolean(runData.feasible))}`}</p>
                <p>{`运行耗时：${runData.runtimeMs != null ? `${Math.round(Number(runData.runtimeMs))} ms` : 'NA'}`}</p>
                <p>{`目标值：${runData.objectiveValue != null ? Number(runData.objectiveValue).toFixed(3) : 'NA'}`}</p>
                <p>{`回退原因：${runData.fallbackReason || '无'}`}</p>
              </div>
            </section>
          ) : null}
        </section>
      </section>

      <section className="workbench-wide">
        <DashboardStatePanel stateKey={dashboardState.key} title={dashboardState.title} description={dashboardState.description} />

        {notice ? (
          <section className={`notice-banner card notice-${noticeTone}`} role="status" aria-live="polite">
            <p>{notice}</p>
          </section>
        ) : null}

        <section className="metrics-grid" aria-label="运行指标">
          <MetricCard
            label="求解耗时"
            value={runData ? formatMs(runData.metrics.solveTimeMs) : '--'}
            hint="本次运行完成速度"
            tooltip="本次运行的总耗时。通常越低表示响应越快。"
          />
          <MetricCard
            label="不可行率"
            value={runData ? formatPct(runData.metrics.infeasibilityRate) : '--'}
            hint="约束被违反的比例"
            tooltip="结果中违反约束的占比，越低通常表示结果越可靠。"
          />
          <MetricCard
            label="次优度"
            value={runData ? formatPct(runData.metrics.suboptimality) : '--'}
            hint="与参考最优目标的差距"
            tooltip="当前目标与参考最佳目标的差距，越低通常代表质量越高。"
          />
        </section>
      </section>

      <section className="card analysis-intro">
        <h2>分析流程</h2>
        <p>
          第一步查看耗时趋势，第二步比较策略成本，第三步阅读完整策略表并解释结果细节。
        </p>
      </section>

      <section className="analysis-grid">
        <section className="card chart-panel">
          <div className="table-head">
            <h2>步骤 1-2：趋势与对比</h2>
            <p>图表优先使用后端返回值，仅在缺失时做本地兜底推导。</p>
          </div>

          <div className="chart-grid">
            <article className="chart-card">
              <h3 className="chart-title">1）求解耗时趋势</h3>
              <p className="chart-copy">用于快速观察近期耗时变化方向（ms）。</p>
              <TrendChart points={trendPoints} unit="ms" />
            </article>

            <article className="chart-card">
              <h3 className="chart-title">2）策略成本对比</h3>
              <p className="chart-copy">展示最新运行快照中各策略的相对成本。</p>
              <ComparisonChart rows={comparisonRows} formatValue={formatComparisonCost} />
            </article>
          </div>
        </section>

        <section className="card table-wrap strategy-panel">
          <div className="table-head">
            <h2>步骤 3：策略表</h2>
            <p>{runData ? `运行 ID：${runData.runId}` : '当前暂无运行结果，请先点击“开始运行”。'}</p>
          </div>

          {runData ? (
            <div className="table-scroll">
              <table>
                <caption className="sr-only">策略输出表</caption>
                <thead>
                  <tr>
                    <th scope="col">策略</th>
                    <th scope="col">可行</th>
                    <th scope="col">成本</th>
                    <th scope="col">排名</th>
                  </tr>
                </thead>
                <tbody>
                  {runData.strategies.map((row) => (
                    <tr key={row.id}>
                      <td>{row.name}</td>
                      <td>{row.feasible ? '是' : '否'}</td>
                      <td>{row.cost.toFixed(3)}</td>
                      <td>{row.rank}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="empty-copy">暂无策略表数据。请选择场景后运行，或刷新加载历史结果。</p>
          )}
        </section>
      </section>
    </div>
  )
}

export default WorkbenchPage
