function DashboardStatePanel({ stateKey, title, description }) {
  return (
    <section className={`card dashboard-state dashboard-state-${stateKey}`} role="status" aria-live="polite">
      <p className="dashboard-state-title">{title}</p>
      <p className="dashboard-state-copy">{description}</p>
    </section>
  )
}

export default DashboardStatePanel
