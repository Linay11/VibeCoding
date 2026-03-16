export const mockScenarios = [
  {
    id: 'portfolio',
    name: '投资组合优化',
    description: '多周期投资配置，考虑风险和交易约束。',
  },
  {
    id: 'control',
    name: '控制集覆盖',
    description: '含切换与负载曲线约束的控制周期优化。',
  },
  {
    id: 'obstacle',
    name: '障碍规避路径',
    description: '含障碍几何边界约束的路径规划优化。',
  },
  {
    id: 'power-118',
    name: '电力 118 节点 SCUC',
    description: 'IEEE 118 节点机组组合与出力调度场景。',
  },
]

function randomInRange(min, max) {
  return min + Math.random() * (max - min)
}

export function buildMockRun(scenarioId) {
  const base = {
    portfolio: { solve: [18, 42], infeas: [0.001, 0.02], subopt: [0.006, 0.03] },
    control: { solve: [25, 75], infeas: [0.004, 0.035], subopt: [0.01, 0.05] },
    obstacle: { solve: [30, 95], infeas: [0.008, 0.06], subopt: [0.02, 0.08] },
  }[scenarioId] ?? { solve: [20, 60], infeas: [0.002, 0.03], subopt: [0.008, 0.04] }

  const solveTimeMs = randomInRange(base.solve[0], base.solve[1])
  const infeasibilityRate = randomInRange(base.infeas[0], base.infeas[1])
  const suboptimality = randomInRange(base.subopt[0], base.subopt[1])
  const trend = [1.2, 1.1, 1.05, 1, 0.95, 0.9].map((ratio, index) => ({
    label: `R-${6 - index}`,
    value: solveTimeMs * ratio,
  }))

  const strategies = [
    { id: 's-1', name: '策略 A', feasible: true, cost: randomInRange(8.2, 9.6), rank: 1 },
    { id: 's-2', name: '策略 B', feasible: true, cost: randomInRange(9.0, 10.3), rank: 2 },
    { id: 's-3', name: '策略 C', feasible: Math.random() > 0.35, cost: randomInRange(9.8, 12.0), rank: 3 },
  ]

  return {
    runId: `run-${scenarioId}-${Date.now()}`,
    scenarioId,
    generatedAt: new Date().toISOString(),
    requestedRunMode: 'exact',
    metrics: {
      solveTimeMs,
      infeasibilityRate,
      suboptimality,
    },
    trend,
    comparison: strategies.map((item) => ({ label: item.name, value: item.cost })),
    strategies,
    solverModeUsed: 'exact',
    mlConfidence: null,
    repairApplied: null,
    fallbackReason: null,
    modelVersion: null,
    featureSchemaVersion: null,
    runtimeMs: solveTimeMs,
    objectiveValue: strategies[0].cost,
    feasible: strategies.every((item) => item.feasible),
    modelPath: null,
    modelLoadStatus: 'not_requested',
  }
}
