import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import WorkbenchPage from './WorkbenchPage.jsx'
import { getLatestRun, getScenarios, runExperiment } from '../services/optimizerApi.js'

vi.mock('../services/optimizerApi.js', () => ({
  getScenarios: vi.fn(),
  getLatestRun: vi.fn(),
  runExperiment: vi.fn(),
}))

function buildRunPayload({
  source = 'api',
  mode = 'compat',
  note = '兼容模式：缺少可选求解器依赖。',
  notice = '',
  noticeTone = 'info',
  errorType = 'none',
  generatedAt = '2026-03-10T10:00:00.000Z',
  data = {},
} = {}) {
  return {
    source,
    mode,
    modeReason: note,
    notice,
    noticeTone,
    errorType,
    data: {
      runId: `run-portfolio-${mode}`,
      scenarioId: 'portfolio',
      generatedAt,
      requestedRunMode: 'exact',
      metrics: {
        solveTimeMs: 36.4,
        infeasibilityRate: 0.01,
        suboptimality: 0.02,
      },
      strategies: [
        {
          id: 'strategy-1',
          name: 'AdapterBaseline',
          feasible: true,
          cost: 8.2,
          rank: 1,
        },
      ],
      trend: [{ label: 'R-1', value: 36.4 }],
      comparison: [{ label: 'AdapterBaseline', value: 8.2 }],
      adapterMode: mode,
      adapterNote: note,
      solverModeUsed: 'exact',
      mlConfidence: null,
      repairApplied: null,
      fallbackReason: null,
      modelVersion: null,
      featureSchemaVersion: null,
      runtimeMs: 36.4,
      objectiveValue: 8.2,
      feasible: true,
      modelLoadStatus: 'not_requested',
      ...data,
    },
  }
}

function createDeferred() {
  let resolve
  const promise = new Promise((resolver) => {
    resolve = resolver
  })
  return { promise, resolve }
}

describe('WorkbenchPage smoke tests', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getScenarios.mockResolvedValue({
      source: 'api',
      data: [
        {
          id: 'portfolio',
          name: '投资组合优化',
          description: '投资组合场景（测试）',
        },
      ],
      notice: '',
      noticeTone: 'info',
    })
    getLatestRun.mockResolvedValue(
      buildRunPayload({
        source: 'fallback',
        mode: 'fallback',
        note: '后端暂无最新运行结果（NOT_FOUND）。',
      }),
    )
    runExperiment.mockResolvedValue(buildRunPayload())
  })

  it('renders result summary correctly for fallback response', async () => {
    render(<WorkbenchPage />)

    await waitFor(() => {
      expect(getLatestRun).toHaveBeenCalledWith('portfolio')
    })

    const summary = await screen.findByRole('region', { name: /运行摘要/i })
    const scoped = within(summary)

    expect(scoped.getByText('投资组合优化')).toBeInTheDocument()
    expect(scoped.getByText('前端回退')).toBeInTheDocument()
    expect(scoped.getByText('回退演示')).toBeInTheDocument()
    expect(scoped.getByText(/后端暂无最新运行结果/i)).toBeInTheDocument()
  })

  it('renders compat mode reason consistently in summary and state panel', async () => {
    const compatReason = '兼容模式：缺少可选求解器依赖。'
    getLatestRun.mockResolvedValueOnce(
      buildRunPayload({
        mode: 'compat',
        note: compatReason,
      }),
    )

    render(<WorkbenchPage />)

    await waitFor(() => {
      expect(getLatestRun).toHaveBeenCalledWith('portfolio')
    })

    const summary = await screen.findByRole('region', { name: /运行摘要/i })
    const scoped = within(summary)

    expect(scoped.getByText('后端 API')).toBeInTheDocument()
    expect(scoped.getByText('兼容模式')).toBeInTheDocument()

    // The same mode reason should appear both in summary and state panel.
    const reasonMatches = await screen.findAllByText((content) => content.includes(compatReason))
    expect(reasonMatches.length).toBeGreaterThanOrEqual(2)

    expect(screen.getByText('后端兼容模式')).toBeInTheDocument()
  })

  it('shows run state transition from running to result on Run Experiment', async () => {
    let resolveRun
    const deferredRun = new Promise((resolve) => {
      resolveRun = resolve
    })
    runExperiment.mockImplementationOnce(() => deferredRun)

    render(<WorkbenchPage />)

    await waitFor(() => {
      expect(getLatestRun).toHaveBeenCalledWith('portfolio')
    })

    const runButton = await screen.findByRole('button', { name: '开始运行' })
    const refreshButton = screen.getByRole('button', { name: '刷新最新结果' })
    expect(runButton).toBeEnabled()
    expect(refreshButton).toBeEnabled()

    fireEvent.click(runButton)

    await waitFor(() => {
      expect(runExperiment).toHaveBeenCalledWith({ scenarioId: 'portfolio', runMode: 'exact' })
    })

    expect(screen.getByRole('button', { name: '运行中...' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '刷新最新结果' })).toBeDisabled()
    expect(screen.getByText('运行进行中')).toBeInTheDocument()

    resolveRun(
      buildRunPayload({
        mode: 'compat',
        note: '兼容模式：缺少可选求解器依赖。',
        generatedAt: '2026-03-10T11:00:00.000Z',
      }),
    )

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '开始运行' })).toBeEnabled()
    })
    expect(screen.getByRole('button', { name: '刷新最新结果' })).toBeEnabled()
    expect(screen.getByText('后端兼容模式')).toBeInTheDocument()
  })

  it('renders consistent fallback messaging for Refresh Latest with 404 no latest', async () => {
    const noLatestReason = '后端暂无最新运行结果（NOT_FOUND）。'
    const noLatestNotice =
      '该场景暂未找到后端最新运行结果，请先执行一次“开始运行”。当前展示回退运行结果。'

    getLatestRun
      .mockResolvedValueOnce(
        buildRunPayload({
          mode: 'compat',
          note: '兼容模式：缺少可选求解器依赖。',
        }),
      )
      .mockResolvedValueOnce(
        buildRunPayload({
          source: 'fallback',
          mode: 'fallback',
          note: noLatestReason,
          notice: noLatestNotice,
          noticeTone: 'info',
          errorType: 'no_latest',
        }),
      )

    render(<WorkbenchPage />)

    await waitFor(() => {
      expect(getLatestRun).toHaveBeenCalledWith('portfolio')
    })

    fireEvent.click(screen.getByRole('button', { name: '刷新最新结果' }))

    await waitFor(() => {
      expect(getLatestRun).toHaveBeenCalledTimes(2)
    })

    const summary = await screen.findByRole('region', { name: /运行摘要/i })
    const scoped = within(summary)
    expect(scoped.getByText('前端回退')).toBeInTheDocument()
    expect(scoped.getByText('回退演示')).toBeInTheDocument()
    expect(scoped.getByText((content) => content.includes(noLatestReason))).toBeInTheDocument()

    expect(screen.getByText('前端回退模式')).toBeInTheDocument()
    expect(screen.getAllByText((content) => content.includes(noLatestReason)).length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText(noLatestNotice)).toBeInTheDocument()
  })

  it('renders consistent fallback messaging for Refresh Latest network failure', async () => {
    const networkReason = '网络请求失败：前端无法连接后端。'
    const networkNotice =
      '网络异常：无法连接后端 API。请检查 VITE_API_BASE_URL、SSH 隧道与后端服务状态。当前展示回退运行结果。'

    getLatestRun
      .mockResolvedValueOnce(
        buildRunPayload({
          mode: 'compat',
          note: '兼容模式：缺少可选求解器依赖。',
        }),
      )
      .mockResolvedValueOnce(
        buildRunPayload({
          source: 'fallback',
          mode: 'fallback',
          note: networkReason,
          notice: networkNotice,
          noticeTone: 'error',
          errorType: 'network',
        }),
      )

    render(<WorkbenchPage />)

    await waitFor(() => {
      expect(getLatestRun).toHaveBeenCalledWith('portfolio')
    })

    fireEvent.click(screen.getByRole('button', { name: '刷新最新结果' }))

    await waitFor(() => {
      expect(getLatestRun).toHaveBeenCalledTimes(2)
    })

    const summary = await screen.findByRole('region', { name: /运行摘要/i })
    const scoped = within(summary)
    expect(scoped.getByText('前端回退')).toBeInTheDocument()
    expect(scoped.getByText('回退演示')).toBeInTheDocument()
    expect(scoped.getByText((content) => content.includes(networkReason))).toBeInTheDocument()

    expect(screen.getByText('前端回退模式')).toBeInTheDocument()
    expect(screen.getAllByText((content) => content.includes(networkReason)).length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText(networkNotice)).toBeInTheDocument()
  })

  it('handles Refresh Latest success flow in compat mode with button and timestamp update', async () => {
    getLatestRun.mockReset()
    const initialCompat = buildRunPayload({
      mode: 'compat',
      note: '兼容模式：走可选求解路径。',
      generatedAt: 'generated-before-compat',
    })
    const refreshedCompat = buildRunPayload({
      mode: 'compat',
      note: '兼容模式：走可选求解路径。',
      generatedAt: 'generated-after-compat',
    })
    const deferred = createDeferred()

    getLatestRun.mockImplementationOnce(() => Promise.resolve(initialCompat))
    getLatestRun.mockImplementationOnce(() => deferred.promise)

    render(<WorkbenchPage />)

    await screen.findByText('generated-before-compat')

    fireEvent.click(screen.getByRole('button', { name: '刷新最新结果' }))
    expect(screen.getByRole('button', { name: '刷新中...' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '开始运行' })).toBeDisabled()

    deferred.resolve(refreshedCompat)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '刷新最新结果' })).toBeEnabled()
    })
    expect(screen.getByRole('button', { name: '开始运行' })).toBeEnabled()
    expect(screen.queryByText('generated-before-compat')).not.toBeInTheDocument()
    expect(screen.getByText('generated-after-compat')).toBeInTheDocument()
    expect(screen.getByText('兼容模式')).toBeInTheDocument()
  })

  it('handles Refresh Latest success flow in real mode with button and timestamp update', async () => {
    getLatestRun.mockReset()
    const initialReal = buildRunPayload({
      mode: 'real',
      note: '真实后端执行已完成。',
      generatedAt: 'generated-before-real',
    })
    const refreshedReal = buildRunPayload({
      mode: 'real',
      note: '真实后端执行已完成。',
      generatedAt: 'generated-after-real',
    })
    const deferred = createDeferred()

    getLatestRun.mockImplementationOnce(() => Promise.resolve(initialReal))
    getLatestRun.mockImplementationOnce(() => deferred.promise)

    render(<WorkbenchPage />)

    await screen.findByText('generated-before-real')

    fireEvent.click(screen.getByRole('button', { name: '刷新最新结果' }))
    expect(screen.getByRole('button', { name: '刷新中...' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '开始运行' })).toBeDisabled()

    deferred.resolve(refreshedReal)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '刷新最新结果' })).toBeEnabled()
    })
    expect(screen.getByRole('button', { name: '开始运行' })).toBeEnabled()
    expect(screen.queryByText('generated-before-real')).not.toBeInTheDocument()
    expect(screen.getByText('generated-after-real')).toBeInTheDocument()
    expect(screen.getByText('真实执行')).toBeInTheDocument()
    expect(screen.getByText('后端真实执行')).toBeInTheDocument()
  })

  it('refreshes latest data, summary, and state panel after scenario switch', async () => {
    const scenarioData = [
      {
        id: 'portfolio',
        name: '投资组合优化',
        description: '投资组合场景（测试）',
      },
      {
        id: 'control',
        name: '控制集覆盖',
        description: '控制集覆盖场景（测试）',
      },
    ]

    getScenarios.mockResolvedValueOnce({
      source: 'api',
      data: scenarioData,
      notice: '',
      noticeTone: 'info',
    })

    getLatestRun.mockReset()
    getLatestRun.mockImplementation((scenarioId) => {
      if (scenarioId === 'portfolio') {
        return Promise.resolve(
          buildRunPayload({
            mode: 'real',
            note: '投资组合真实运行成功。',
            generatedAt: 'generated-portfolio',
          }),
        )
      }
      if (scenarioId === 'control') {
        return Promise.resolve(
          buildRunPayload({
            mode: 'compat',
            note: '控制集覆盖使用兼容模式（可选依赖缺失）。',
            generatedAt: 'generated-control',
            data: {
              scenarioId: 'control',
              generatedAt: 'generated-control',
            },
          }),
        )
      }
      return Promise.resolve(buildRunPayload())
    })

    render(<WorkbenchPage />)

    await waitFor(() => {
      expect(getLatestRun).toHaveBeenCalledWith('portfolio')
    })
    const initialSummary = await screen.findByRole('region', { name: /运行摘要/i })
    expect(within(initialSummary).getByText('投资组合优化')).toBeInTheDocument()
    expect(screen.getByText('真实执行')).toBeInTheDocument()
    expect(screen.getByText('后端真实执行')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('场景'), { target: { value: 'control' } })

    await waitFor(() => {
      expect(getLatestRun).toHaveBeenCalledWith('control')
    })

    const switchedSummary = await screen.findByRole('region', { name: /运行摘要/i })
    expect(within(switchedSummary).getByText('控制集覆盖')).toBeInTheDocument()
    expect(screen.getByText('兼容模式')).toBeInTheDocument()
    expect(screen.getByText('后端兼容模式')).toBeInTheDocument()
    expect(screen.getAllByText((content) => content.includes('控制集覆盖使用兼容模式（可选依赖缺失）。')).length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('generated-control')).toBeInTheDocument()
  })

  it('keeps the last selected scenario state when an older latest request resolves late', async () => {
    const scenarioData = [
      {
        id: 'portfolio',
        name: '投资组合优化',
        description: '投资组合场景（测试）',
      },
      {
        id: 'control',
        name: '控制集覆盖',
        description: '控制集覆盖场景（测试）',
      },
    ]
    const slowOldRequest = createDeferred()
    const fastNewRequest = createDeferred()
    const controlReason = '控制集覆盖从 latest 端点返回兼容结果。'
    const latePortfolioReason = '延迟到达的投资组合响应应被忽略。'

    getScenarios.mockResolvedValueOnce({
      source: 'api',
      data: scenarioData,
      notice: '',
      noticeTone: 'info',
    })

    getLatestRun.mockReset()
    getLatestRun.mockImplementation((scenarioId) => {
      if (scenarioId === 'portfolio') {
        return slowOldRequest.promise
      }
      if (scenarioId === 'control') {
        return fastNewRequest.promise
      }
      return Promise.resolve(buildRunPayload())
    })

    render(<WorkbenchPage />)

    await waitFor(() => {
      expect(getLatestRun).toHaveBeenCalledWith('portfolio')
    })

    fireEvent.change(screen.getByLabelText('场景'), { target: { value: 'control' } })

    await waitFor(() => {
      expect(getLatestRun).toHaveBeenCalledWith('control')
    })

    fastNewRequest.resolve(
      buildRunPayload({
        mode: 'compat',
        note: controlReason,
        generatedAt: 'generated-control-fast',
      }),
    )

    await waitFor(() => {
      expect(screen.getByText('generated-control-fast')).toBeInTheDocument()
    })

    slowOldRequest.resolve(
      buildRunPayload({
        mode: 'real',
        note: latePortfolioReason,
        generatedAt: 'generated-portfolio-late',
      }),
    )

    await new Promise((resolve) => setTimeout(resolve, 0))

    const finalSummary = screen.getByRole('region', { name: /运行摘要/i })
    const scoped = within(finalSummary)
    expect(scoped.getByText('控制集覆盖')).toBeInTheDocument()
    expect(scoped.getByText('兼容模式')).toBeInTheDocument()
    expect(scoped.getByText((content) => content.includes(controlReason))).toBeInTheDocument()
    expect(scoped.getByText('generated-control-fast')).toBeInTheDocument()
    expect(screen.getByText('后端兼容模式')).toBeInTheDocument()
    expect(screen.queryByText('generated-portfolio-late')).not.toBeInTheDocument()
    expect(screen.queryByText(latePortfolioReason)).not.toBeInTheDocument()
    expect(screen.queryByText('后端真实执行')).not.toBeInTheDocument()
  })

  it('shows requested versus actual mode and diagnostics for power-118', async () => {
    getScenarios.mockResolvedValueOnce({
      source: 'api',
      data: [
        {
          id: 'power-118',
          name: '电力 118 节点 SCUC',
          description: '电力场景（测试）',
        },
      ],
      notice: '',
      noticeTone: 'info',
    })
    getLatestRun.mockResolvedValueOnce(
      buildRunPayload({
        mode: 'real',
        note: 'Power118 请求 hybrid，最终回退 exact。',
        data: {
          scenarioId: 'power-118',
          requestedRunMode: 'ml',
          solverModeUsed: 'exact',
          fallbackReason: 'ml model unavailable: artifact missing',
          modelVersion: 'power118-baseline-v1',
          featureSchemaVersion: 'power118-feature-schema-v1',
          mlConfidence: 0.74,
          repairApplied: true,
          runtimeMs: 128.0,
          objectiveValue: 456.7,
          feasible: true,
          modelLoadStatus: 'failed',
        },
      }),
    )

    render(<WorkbenchPage />)

    await waitFor(() => {
      expect(getLatestRun).toHaveBeenCalledWith('power-118')
    })

    expect(screen.getByLabelText('电力 118 诊断信息')).toBeInTheDocument()
    expect(screen.getByText('请求模式：ml')).toBeInTheDocument()
    expect(screen.getByText('实际模式：exact')).toBeInTheDocument()
    expect(screen.getByText('回退原因：ml model unavailable: artifact missing')).toBeInTheDocument()
    expect(screen.getByText(/请求模式 ml，实际使用 exact/i)).toBeInTheDocument()
  })
})
