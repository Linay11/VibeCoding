import { buildMockRun, mockScenarios } from './mockData.js'

const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000'
const RAW_API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? import.meta.env.VITE_API_BASE ?? ''
const API_BASE_URL = String(RAW_API_BASE_URL || '')
  .trim()
  .replace(/\/+$/, '') || DEFAULT_API_BASE_URL

function buildApiUrl(path) {
  const pathText = String(path ?? '')
  const normalizedPath = pathText.startsWith('/') ? pathText : `/${pathText}`
  return `${API_BASE_URL}${normalizedPath}`
}

function resolveTimeoutMs(rawValue) {
  const parsed = Number(rawValue)
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return 60000
  }
  return Math.round(parsed)
}

const DEFAULT_REQUEST_TIMEOUT_MS = resolveTimeoutMs(import.meta.env.VITE_API_TIMEOUT_MS ?? 60000)
const POWER118_EXACT_TIMEOUT_MS = 120000

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
  const { timeoutMs, headers, signal, ...fetchOptions } = options
  const effectiveTimeoutMs = resolveTimeoutMs(timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS)
  const timeoutController = new AbortController()
  const timeoutHandle = setTimeout(() => {
    timeoutController.abort()
  }, effectiveTimeoutMs)
  let detachSignal = null
  if (signal && typeof signal.addEventListener === 'function') {
    if (signal.aborted) {
      timeoutController.abort()
    } else {
      const onAbort = () => {
        timeoutController.abort()
      }
      signal.addEventListener('abort', onAbort, { once: true })
      detachSignal = () => {
        signal.removeEventListener('abort', onAbort)
      }
    }
  }

  let response
  try {
    response = await fetch(buildApiUrl(path), {
      ...fetchOptions,
      headers: {
        'Content-Type': 'application/json',
        ...(headers ?? {}),
      },
      signal: timeoutController.signal,
    })
  } catch (error) {
    const abortedByExternalSignal = Boolean(signal?.aborted)
    const timedOut = timeoutController.signal.aborted && !abortedByExternalSignal
    if (timedOut) {
      throw makeClientError('timeout', `Backend API request timed out after ${effectiveTimeoutMs} ms.`, {
        cause: error,
        timeoutMs: effectiveTimeoutMs,
      })
    }
    if (abortedByExternalSignal) {
      throw makeClientError('aborted', 'Backend API request was cancelled.', {
        cause: error,
      })
    }
    throw makeClientError('network', 'Cannot reach backend API', { cause: error })
  } finally {
    clearTimeout(timeoutHandle)
    if (typeof detachSignal === 'function') {
      detachSignal()
    }
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
          description: `场景 ${index + 1}`,
        }
      }

      return {
        id: item.id ?? item.name ?? `scenario-${index + 1}`,
        name: item.name ?? item.id ?? `场景 ${index + 1}`,
        description: item.description ?? '优化实验场景',
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
    requestedRunMode: raw.requestedRunMode ?? null,
    metrics,
    strategies: normalizedStrategies,
    trend: normalizedTrend,
    comparison: normalizedComparison,
    adapterMode,
    solverModeUsed: raw.solverModeUsed ?? '',
    mlConfidence: raw.mlConfidence ?? null,
    repairApplied: raw.repairApplied ?? null,
    fallbackReason: raw.fallbackReason ?? null,
    modelVersion: raw.modelVersion ?? null,
    featureSchemaVersion: raw.featureSchemaVersion ?? null,
    runtimeMs: Math.max(0, toNumber(raw.runtimeMs ?? raw.metrics?.solveTimeMs ?? 0, 0)),
    objectiveValue:
      raw.objectiveValue != null
        ? toNumber(raw.objectiveValue, 0)
        : normalizedStrategies[0]
          ? normalizedStrategies[0].cost
          : null,
    feasible: raw.feasible != null ? Boolean(raw.feasible) : normalizedStrategies.some((row) => row.feasible),
    modelPath: raw.modelPath ?? null,
    modelLoadStatus: raw.modelLoadStatus ?? null,
    adapterNote:
      adapterNote ||
      (adapterMode === 'real'
        ? '后端真实执行已完成。'
        : adapterMode === 'compat'
          ? '当前为后端适配器兼容模式结果。'
          : '当前展示的是前端回退数据。'),
  }
}

function classifyApiFailure(error, operation) {
  if (error?.kind === 'timeout') {
    const timeoutMs = Number(error.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS)
    const timeoutText = Number.isFinite(timeoutMs) ? `${Math.round(timeoutMs)} ms` : 'configured timeout'
    return {
      type: 'timeout',
      reason: `网络请求超时（${timeoutText}）。`,
      userMessage: '后端请求超时。请检查后端负载、隧道延迟，或适当提高超时配置。',
      tone: 'warning',
    }
  }

  if (error?.kind === 'aborted') {
    return {
      type: 'cancelled',
      reason: '请求在完成前被取消。',
      userMessage: '请求在完成前已被取消，如有需要可重新发起。',
      tone: 'info',
    }
  }

  if (error?.kind === 'network') {
    return {
      type: 'network',
      reason: '网络请求失败：前端无法连接后端。',
      userMessage:
        '网络异常：无法连接后端 API。请检查 VITE_API_BASE_URL、SSH 隧道与后端服务状态。',
      tone: 'error',
    }
  }

  if (error?.kind === 'http') {
    const code = error.code ?? `HTTP_${error.status ?? 'UNKNOWN'}`
    const backendMessage = String(error.message ?? '').trim()

    if (operation === 'latest' && error.status === 404 && code === 'NOT_FOUND') {
      return {
        type: 'no_latest',
        reason: `后端暂无最新运行结果（${code}）。`,
        userMessage: '该场景暂未找到后端最新运行结果，请先执行一次“开始运行”。',
        tone: 'info',
      }
    }

    if ((error.status ?? 0) >= 500) {
      return {
        type: 'backend_failed',
        reason: `${code}${backendMessage ? `: ${backendMessage}` : ''}`,
        userMessage:
          '后端运行失败。界面已切换为前端回退数据，便于你继续操作与排查。',
        tone: 'error',
      }
    }

    return {
      type: 'request_invalid',
      reason: `${code}${backendMessage ? `: ${backendMessage}` : ''}`,
      userMessage: '后端拒绝了本次请求，请检查场景选择与适配器参数。',
      tone: 'warning',
    }
  }

  return {
    type: 'unknown',
    reason: `未预期异常：${String(error)}`,
    userMessage: '发生未预期的接口异常，当前已切换为前端回退数据。',
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
      notice: `${failure.userMessage} 当前展示示例场景。`,
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
      notice: `${failure.userMessage} 当前展示回退运行结果。`,
      noticeTone: failure.tone,
      errorType: failure.type,
    }
  }
}

export async function runExperiment({ scenarioId, runMode = 'exact', timeLimitMs = null, fallbackToExact = true }) {
  try {
    const requestTimeoutMs =
      scenarioId === 'power-118' && runMode === 'exact'
        ? Math.max(DEFAULT_REQUEST_TIMEOUT_MS, POWER118_EXACT_TIMEOUT_MS)
        : DEFAULT_REQUEST_TIMEOUT_MS

    const payload = await requestJson('/api/runs', {
      method: 'POST',
      body: JSON.stringify({ scenarioId, runMode, timeLimitMs, fallbackToExact }),
      timeoutMs: requestTimeoutMs,
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
      notice: `${failure.userMessage} 浏览器端已生成回退运行结果。`,
      noticeTone: failure.tone,
      errorType: failure.type,
    }
  }
}
