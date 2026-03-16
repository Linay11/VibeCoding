import { memo } from 'react'

function DashboardStatePanel({ stateKey, title, description }) {
  const safeStateKey = stateKey || 'empty'
  const safeTitle = title || '状态等待中'
  const safeDescription = description || '正在等待更新。'
  return (
    <section className={`card dashboard-state dashboard-state-${safeStateKey}`} role="status" aria-live="polite">
      <p className="dashboard-state-title">{safeTitle}</p>
      <p className="dashboard-state-copy">{safeDescription}</p>
    </section>
  )
}

export default memo(DashboardStatePanel)
