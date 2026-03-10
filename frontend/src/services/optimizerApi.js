import { buildMockRun, mockScenarios } from './mockData.js'

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms)
  })
}

function toNumber(value, fallback = 0) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function makeClientError(kind, message, extra = {}) {
  const error = new Error(message)
  error.kind = kind
  Object.assign(error, extra)
  return error
}

async function requestJson(path, options = {}) {
  let response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers ?? {}),
      },
      ...options,
    })
  } catch (error) {
    throw makeClientError('network', 'Cannot reach backend API', { cause: error })
  }

  let payload = null
  try {
    payload = await response.json()
  } catch {
    payload = null
  }

  if (!response.ok) {
    const errorCode = payload?.error?.code ?? `HTTP_${response.status}`
    const errorMessage = payload?.error?.message ?? `Request failed (${response.status})`
    throw makeClientError('http', errorMessage, {
      status: response.status,
      code: errorCode,
      payload,
    })
  }

  return payload
}

function normalizeScenarios(payload) {
  const raw = payload?.scenarios ?? payload?.data ?? payload
  if (!Array.isArray(raw)) {
    return []
  }

  return raw
    .map((item, index) => {
      if (typeof item === 'string') {
        return {
          id: item,
          name: item,
          description: `Scenario ${index + 1}`,
        }
      }

      return {
        id: item.id ?? item.name ?? `scenario-${index + 1}`,
        name: item.name ?? item.id ?? `Scenario ${index + 1}`,
        description: item.description ?? 'Optimization scenario',
      }
    })
    .filter((item) => item.id)
}

function normalizeRun(payload, scenarioId, modeOverride = null, noteOverride = '') {
  const raw = payload?.run ?? payload?.data ?? payload
  if (!raw || typeof raw !== 'object') {
    return null
  }

  const metrics = {
    solveTimeMs: Math.max(0, toNumber(raw.metrics?.solveTimeMs ?? raw.solveTimeMs ?? raw.timeMs ?? 0, 0)),
    infeasibilityRate: Math.min(Math.max(toNumber(raw.metrics?.infeasibilityRate ?? raw.infeasibilityRate ?? 0, 0), 0), 1),
    suboptimality: Math.max(0, toNumber(raw.metrics?.suboptimality ?? raw.suboptimality ?? 0, 0)),
  }

  const normalizedStrategies = Array.isArray(raw.strategies)
    ? raw.strategies.map((row, index) => ({
        id: row.id ?? `strategy-${index + 1}`,
        name: row.name ?? `Strategy ${index + 1}`,
        feasible: Boolean(row.feasible),
        cost: toNumber(row.cost ?? 0, 0),
        rank: Math.max(1, toNumber(row.rank ?? index + 1, index + 1)),
      }))
    : []

  const normalizedTrend = Array.isArray(raw.trend)
    ? raw.trend
        .map((item, index) => ({
          label: item?.label ?? `R${index + 1}`,
          value: Math.max(0, toNumber(item?.value ?? item?.solveTimeMs ?? item?.solve ?? 0, 0)),
        }))
        .filter((item) => Number.isFinite(item.value))
    : []

  const normalizedComparison = Array.isArray(raw.comparison ?? raw.comparisons)
    ? (raw.comparison ?? raw.comparisons)
        .map((item, index) => ({
          label: item?.label ?? `Item ${index + 1}`,
          value: toNumber(item?.value ?? item?.cost ?? 0, 0),
        }))
        .filter((item) => Number.isFinite(item.value))
    : []

  const modeFromPayload = String(raw.adapterMode ?? '').toLowerCase()
  const adapterMode =
    modeOverride ?? (modeFromPayload === 'real' || modeFromPayload === 'compat' ? modeFromPayload : 'compat')
  const adapterNote = String(noteOverride || raw.adapterNote || '').trim()

  return {
    runId: raw.runId ?? raw.id ?? `run-${scenarioId}-${Date.now()}`,
    scenarioId: raw.scenarioId ?? scenarioId,
    generatedAt: raw.generatedAt ?? new Date().toISOString(),
    metrics,
    strategies: normalizedStrategies,
    trend: normalizedTrend,
    comparison: normalizedComparison,
    adapterMode,
    adapterNote:
      adapterNote ||
      (adapterMode === 'real'
        ? 'Real backend execution completed.'
        : adapterMode === 'compat'
          ? 'Compatibility mode response from backend adapter.'
          : 'Frontend fallback data is currently displayed.'),
  }
}

function classifyApiFailure(error, operation) {
  if (error?.kind === 'network') {
    return {
      type: 'network',
      reason: 'Network request failed: backend is unreachable from frontend.',
      userMessage:
        'Network failure: cannot reach backend API. Check VITE_API_BASE, SSH tunnel status, and backend service health.',
      tone: 'error',
    }
  }

  if (error?.kind === 'http') {
    const code = error.code ?? `HTTP_${error.status ?? 'UNKNOWN'}`
    const backendMessage = String(error.message ?? '').trim()

    if (operation === 'latest' && error.status === 404 && code === 'NOT_FOUND') {
      return {
        type: 'no_latest',
        reason: `No latest run exists on backend yet (${code}).`,
        userMessage: 'No latest backend run found for this scenario yet. Run Experiment once to create it.',
        tone: 'info',
      }
    }

    if ((error.status ?? 0) >= 500) {
      return {
        type: 'backend_failed',
        reason: `${code}${backendMessage ? `: ${backendMessage}` : ''}`,
        userMessage:
          'Backend run failed. The UI switched to fallback demo data so you can keep exploring without blocking.',
        tone: 'error',
      }
    }

    return {
      type: 'request_invalid',
      reason: `${code}${backendMessage ? `: ${backendMessage}` : ''}`,
      userMessage: 'Backend rejected the request. Please verify scenario selection and backend adapter parameters.',
      tone: 'warning',
    }
  }

  return {
    type: 'unknown',
    reason: `Unexpected error: ${String(error)}`,
    userMessage: 'Unexpected API failure occurred. Fallback demo data is now displayed.',
    tone: 'warning',
  }
}

function buildFallbackRun(scenarioId, reason) {
  return normalizeRun(buildMockRun(scenarioId), scenarioId, 'fallback', reason)
}

export async function getScenarios() {
  try {
    const payload = await requestJson('/api/scenarios')
    const normalized = normalizeScenarios(payload)
    if (normalized.length === 0) {
      throw makeClientError('http', 'No scenarios from API', {
        status: 500,
        code: 'INVALID_RESPONSE',
      })
    }
    return { source: 'api', data: normalized, notice: '', noticeTone: 'info' }
  } catch (error) {
    const failure = classifyApiFailure(error, 'scenarios')
    await delay(200)
    return {
      source: 'fallback',
      data: mockScenarios,
      notice: `${failure.userMessage} Showing demo scenarios.`,
      noticeTone: failure.tone,
      mode: 'fallback',
      modeReason: failure.reason,
      errorType: failure.type,
    }
  }
}

export async function getLatestRun(scenarioId) {
  try {
    const payload = await requestJson(`/api/runs/latest?scenarioId=${encodeURIComponent(scenarioId)}`)
    const normalized = normalizeRun(payload, scenarioId)
    if (!normalized) {
      throw makeClientError('http', 'No latest run from API', {
        status: 500,
        code: 'INVALID_RESPONSE',
      })
    }
    return {
      source: 'api',
      mode: normalized.adapterMode,
      modeReason: normalized.adapterNote,
      data: normalized,
      notice: '',
      noticeTone: 'info',
      errorType: 'none',
    }
  } catch (error) {
    const failure = classifyApiFailure(error, 'latest')
    await delay(220)
    return {
      source: 'fallback',
      mode: 'fallback',
      modeReason: failure.reason,
      data: buildFallbackRun(scenarioId, failure.reason),
      notice: `${failure.userMessage} Showing fallback run data.`,
      noticeTone: failure.tone,
      errorType: failure.type,
    }
  }
}

export async function runExperiment({ scenarioId }) {
  try {
    const payload = await requestJson('/api/runs', {
      method: 'POST',
      body: JSON.stringify({ scenarioId }),
    })
    const normalized = normalizeRun(payload, scenarioId)
    if (!normalized) {
      throw makeClientError('http', 'Run API returned invalid payload', {
        status: 500,
        code: 'INVALID_RESPONSE',
      })
    }
    return {
      source: 'api',
      mode: normalized.adapterMode,
      modeReason: normalized.adapterNote,
      data: normalized,
      notice: '',
      noticeTone: 'info',
      errorType: 'none',
    }
  } catch (error) {
    const failure = classifyApiFailure(error, 'run')
    await delay(320)
    return {
      source: 'fallback',
      mode: 'fallback',
      modeReason: failure.reason,
      data: buildFallbackRun(scenarioId, failure.reason),
      notice: `${failure.userMessage} A fallback run has been generated in the browser.`,
      noticeTone: failure.tone,
      errorType: failure.type,
    }
  }
}
