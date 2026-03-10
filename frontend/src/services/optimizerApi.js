import { buildMockRun, mockScenarios } from './mockData.js'

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms)
  })
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers ?? {}),
    },
    ...options,
  })

  if (!response.ok) {
    throw new Error(`Request failed (${response.status})`)
  }

  return response.json()
}

function readableError(error) {
  if (!error) {
    return 'Unknown error'
  }
  if (error instanceof Error) {
    return error.message
  }
  return String(error)
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

function normalizeRun(payload, scenarioId) {
  const raw = payload?.run ?? payload?.data ?? payload
  if (!raw || typeof raw !== 'object') {
    return null
  }

  const normalizedTrend = Array.isArray(raw.trend)
    ? raw.trend
        .map((item, index) => ({
          label: item?.label ?? `R${index + 1}`,
          value: Number(item?.value ?? item?.solveTimeMs ?? item?.solve ?? 0),
        }))
        .filter((item) => Number.isFinite(item.value))
    : []

  const normalizedComparison = Array.isArray(raw.comparison ?? raw.comparisons)
    ? (raw.comparison ?? raw.comparisons)
        .map((item, index) => ({
          label: item?.label ?? `Item ${index + 1}`,
          value: Number(item?.value ?? item?.cost ?? 0),
        }))
        .filter((item) => Number.isFinite(item.value))
    : []

  return {
    runId: raw.runId ?? raw.id ?? `run-${scenarioId}-${Date.now()}`,
    scenarioId: raw.scenarioId ?? scenarioId,
    generatedAt: raw.generatedAt ?? new Date().toISOString(),
    metrics: {
      solveTimeMs: raw.metrics?.solveTimeMs ?? raw.solveTimeMs ?? raw.timeMs ?? 0,
      infeasibilityRate: raw.metrics?.infeasibilityRate ?? raw.infeasibilityRate ?? 0,
      suboptimality: raw.metrics?.suboptimality ?? raw.suboptimality ?? 0,
    },
    strategies: Array.isArray(raw.strategies)
      ? raw.strategies.map((row, index) => ({
          id: row.id ?? `strategy-${index + 1}`,
          name: row.name ?? `Strategy ${index + 1}`,
          feasible: Boolean(row.feasible),
          cost: Number(row.cost ?? 0),
          rank: Number(row.rank ?? index + 1),
        }))
      : [],
    trend: normalizedTrend,
    comparison: normalizedComparison,
  }
}

export async function getScenarios() {
  try {
    const payload = await requestJson('/api/scenarios')
    const normalized = normalizeScenarios(payload)
    if (normalized.length === 0) {
      throw new Error('No scenarios from API')
    }
    return { source: 'api', data: normalized, notice: '' }
  } catch (error) {
    await delay(250)
    return {
      source: 'mock',
      data: mockScenarios,
      notice: `Backend scenario API unavailable (${readableError(error)}). Showing demo scenarios.`,
    }
  }
}

export async function getLatestRun(scenarioId) {
  try {
    const payload = await requestJson(`/api/runs/latest?scenarioId=${encodeURIComponent(scenarioId)}`)
    const normalized = normalizeRun(payload, scenarioId)
    if (!normalized) {
      throw new Error('No latest run from API')
    }
    return { source: 'api', data: normalized, notice: '' }
  } catch (error) {
    await delay(220)
    return {
      source: 'mock',
      data: buildMockRun(scenarioId),
      notice: `Latest run API unavailable (${readableError(error)}). Showing mock run data.`,
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
      throw new Error('Run API did not return valid result')
    }
    return { source: 'api', data: normalized, notice: '' }
  } catch (error) {
    await delay(450)
    return {
      source: 'mock',
      data: buildMockRun(scenarioId),
      notice: `Run API unavailable (${readableError(error)}). Generated a mock run instead.`,
    }
  }
}
