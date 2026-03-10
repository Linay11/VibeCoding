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
  note = 'Compatibility mode due to missing optional solver dependency.',
  notice = '',
  noticeTone = 'info',
  errorType = 'none',
  generatedAt = '2026-03-10T10:00:00.000Z',
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
          name: 'Portfolio Optimization',
          description: 'Portfolio scenario for smoke test',
        },
      ],
      notice: '',
      noticeTone: 'info',
    })
    getLatestRun.mockResolvedValue(
      buildRunPayload({
        source: 'fallback',
        mode: 'fallback',
        note: 'No latest run exists on backend yet (NOT_FOUND).',
      }),
    )
    runExperiment.mockResolvedValue(buildRunPayload())
  })

  it('renders result summary correctly for fallback response', async () => {
    render(<WorkbenchPage />)

    await waitFor(() => {
      expect(getLatestRun).toHaveBeenCalledWith('portfolio')
    })

    const summary = await screen.findByRole('region', { name: /run summary/i })
    const scoped = within(summary)

    expect(scoped.getByText('Portfolio Optimization')).toBeInTheDocument()
    expect(scoped.getByText('Frontend fallback')).toBeInTheDocument()
    expect(scoped.getByText('Fallback demo')).toBeInTheDocument()
    expect(scoped.getByText(/No latest run exists on backend yet/i)).toBeInTheDocument()
  })

  it('renders compat mode reason consistently in summary and state panel', async () => {
    const compatReason = 'Compatibility mode due to missing optional solver dependency.'
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

    const summary = await screen.findByRole('region', { name: /run summary/i })
    const scoped = within(summary)

    expect(scoped.getByText('Backend API')).toBeInTheDocument()
    expect(scoped.getByText('Compat run')).toBeInTheDocument()

    // The same mode reason should appear both in summary and state panel.
    const reasonMatches = await screen.findAllByText(compatReason)
    expect(reasonMatches.length).toBeGreaterThanOrEqual(2)

    expect(screen.getByText('Backend compatibility mode')).toBeInTheDocument()
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

    const runButton = await screen.findByRole('button', { name: 'Run Experiment' })
    const refreshButton = screen.getByRole('button', { name: 'Refresh Latest' })
    expect(runButton).toBeEnabled()
    expect(refreshButton).toBeEnabled()

    fireEvent.click(runButton)

    await waitFor(() => {
      expect(runExperiment).toHaveBeenCalledWith({ scenarioId: 'portfolio' })
    })

    expect(screen.getByRole('button', { name: 'Running...' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Refresh Latest' })).toBeDisabled()
    expect(screen.getByText('Run in progress')).toBeInTheDocument()

    resolveRun(
      buildRunPayload({
        mode: 'compat',
        note: 'Compatibility mode due to missing optional solver dependency.',
        generatedAt: '2026-03-10T11:00:00.000Z',
      }),
    )

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Run Experiment' })).toBeEnabled()
    })
    expect(screen.getByRole('button', { name: 'Refresh Latest' })).toBeEnabled()
    expect(screen.getByText('Backend compatibility mode')).toBeInTheDocument()
  })

  it('renders consistent fallback messaging for Refresh Latest with 404 no latest', async () => {
    const noLatestReason = 'No latest run exists on backend yet (NOT_FOUND).'
    const noLatestNotice =
      'No latest backend run found for this scenario yet. Run Experiment once to create it. Showing fallback run data.'

    getLatestRun
      .mockResolvedValueOnce(
        buildRunPayload({
          mode: 'compat',
          note: 'Compatibility mode due to missing optional solver dependency.',
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

    fireEvent.click(screen.getByRole('button', { name: 'Refresh Latest' }))

    await waitFor(() => {
      expect(getLatestRun).toHaveBeenCalledTimes(2)
    })

    const summary = await screen.findByRole('region', { name: /run summary/i })
    const scoped = within(summary)
    expect(scoped.getByText('Frontend fallback')).toBeInTheDocument()
    expect(scoped.getByText('Fallback demo')).toBeInTheDocument()
    expect(scoped.getByText(noLatestReason)).toBeInTheDocument()

    expect(screen.getByText('Frontend fallback mode')).toBeInTheDocument()
    expect(screen.getAllByText(noLatestReason).length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText(noLatestNotice)).toBeInTheDocument()
  })

  it('renders consistent fallback messaging for Refresh Latest network failure', async () => {
    const networkReason = 'Network request failed: backend is unreachable from frontend.'
    const networkNotice =
      'Network failure: cannot reach backend API. Check VITE_API_BASE, SSH tunnel status, and backend service health. Showing fallback run data.'

    getLatestRun
      .mockResolvedValueOnce(
        buildRunPayload({
          mode: 'compat',
          note: 'Compatibility mode due to missing optional solver dependency.',
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

    fireEvent.click(screen.getByRole('button', { name: 'Refresh Latest' }))

    await waitFor(() => {
      expect(getLatestRun).toHaveBeenCalledTimes(2)
    })

    const summary = await screen.findByRole('region', { name: /run summary/i })
    const scoped = within(summary)
    expect(scoped.getByText('Frontend fallback')).toBeInTheDocument()
    expect(scoped.getByText('Fallback demo')).toBeInTheDocument()
    expect(scoped.getByText(networkReason)).toBeInTheDocument()

    expect(screen.getByText('Frontend fallback mode')).toBeInTheDocument()
    expect(screen.getAllByText(networkReason).length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText(networkNotice)).toBeInTheDocument()
  })

  it('handles Refresh Latest success flow in compat mode with button and timestamp update', async () => {
    getLatestRun.mockReset()
    const initialCompat = buildRunPayload({
      mode: 'compat',
      note: 'Compat mode with optional solver path.',
      generatedAt: 'generated-before-compat',
    })
    const refreshedCompat = buildRunPayload({
      mode: 'compat',
      note: 'Compat mode with optional solver path.',
      generatedAt: 'generated-after-compat',
    })
    const deferred = createDeferred()

    getLatestRun.mockImplementationOnce(() => Promise.resolve(initialCompat))
    getLatestRun.mockImplementationOnce(() => deferred.promise)

    render(<WorkbenchPage />)

    await screen.findByText('generated-before-compat')

    fireEvent.click(screen.getByRole('button', { name: 'Refresh Latest' }))
    expect(screen.getByRole('button', { name: 'Refreshing...' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Run Experiment' })).toBeDisabled()

    deferred.resolve(refreshedCompat)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Refresh Latest' })).toBeEnabled()
    })
    expect(screen.getByRole('button', { name: 'Run Experiment' })).toBeEnabled()
    expect(screen.queryByText('generated-before-compat')).not.toBeInTheDocument()
    expect(screen.getByText('generated-after-compat')).toBeInTheDocument()
    expect(screen.getByText('Compat run')).toBeInTheDocument()
  })

  it('handles Refresh Latest success flow in real mode with button and timestamp update', async () => {
    getLatestRun.mockReset()
    const initialReal = buildRunPayload({
      mode: 'real',
      note: 'Real backend execution completed.',
      generatedAt: 'generated-before-real',
    })
    const refreshedReal = buildRunPayload({
      mode: 'real',
      note: 'Real backend execution completed.',
      generatedAt: 'generated-after-real',
    })
    const deferred = createDeferred()

    getLatestRun.mockImplementationOnce(() => Promise.resolve(initialReal))
    getLatestRun.mockImplementationOnce(() => deferred.promise)

    render(<WorkbenchPage />)

    await screen.findByText('generated-before-real')

    fireEvent.click(screen.getByRole('button', { name: 'Refresh Latest' }))
    expect(screen.getByRole('button', { name: 'Refreshing...' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Run Experiment' })).toBeDisabled()

    deferred.resolve(refreshedReal)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Refresh Latest' })).toBeEnabled()
    })
    expect(screen.getByRole('button', { name: 'Run Experiment' })).toBeEnabled()
    expect(screen.queryByText('generated-before-real')).not.toBeInTheDocument()
    expect(screen.getByText('generated-after-real')).toBeInTheDocument()
    expect(screen.getByText('Real run')).toBeInTheDocument()
    expect(screen.getByText('Backend real execution')).toBeInTheDocument()
  })

  it('refreshes latest data, summary, and state panel after scenario switch', async () => {
    const scenarioData = [
      {
        id: 'portfolio',
        name: 'Portfolio Optimization',
        description: 'Portfolio scenario for smoke test',
      },
      {
        id: 'control',
        name: 'Control Setcover',
        description: 'Control scenario for smoke test',
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
            note: 'Portfolio real run succeeded.',
            generatedAt: 'generated-portfolio',
          }),
        )
      }
      if (scenarioId === 'control') {
        return Promise.resolve(
          buildRunPayload({
            mode: 'compat',
            note: 'Control compat run due to optional dependency gap.',
            generatedAt: 'generated-control',
            data: {
              ...buildRunPayload().data,
              scenarioId: 'control',
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
    const initialSummary = await screen.findByRole('region', { name: /run summary/i })
    expect(within(initialSummary).getByText('Portfolio Optimization')).toBeInTheDocument()
    expect(screen.getByText('Real run')).toBeInTheDocument()
    expect(screen.getByText('Backend real execution')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('Scenario'), { target: { value: 'control' } })

    await waitFor(() => {
      expect(getLatestRun).toHaveBeenCalledWith('control')
    })

    const switchedSummary = await screen.findByRole('region', { name: /run summary/i })
    expect(within(switchedSummary).getByText('Control Setcover')).toBeInTheDocument()
    expect(screen.getByText('Compat run')).toBeInTheDocument()
    expect(screen.getByText('Backend compatibility mode')).toBeInTheDocument()
    expect(screen.getAllByText('Control compat run due to optional dependency gap.').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('generated-control')).toBeInTheDocument()
  })
})
