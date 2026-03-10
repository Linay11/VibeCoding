import { Link } from 'react-router-dom'

const highlights = [
  {
    title: 'Experiment-First Workflow',
    description:
      'Move from scenario selection to run output, trend analysis, and strategy explanation in one page.',
  },
  {
    title: 'Transparent Runtime Modes',
    description:
      'Instantly see whether a result is real backend execution, adapter compat mode, or frontend fallback.',
  },
  {
    title: 'Resilient Demo Experience',
    description:
      'If endpoints are unavailable, the dashboard stays explorable with explicit fallback indicators.',
  },
]

function HomePage() {
  return (
    <div className="fade-in">
      <section className="hero card">
        <p className="eyebrow">Optimization Experiment Dashboard</p>
        <h1>Analyze optimization runs with clear, explainable runtime context.</h1>
        <p className="lead">
          This frontend dashboard wraps your optimizer workflows with structured summaries, stable
          run diagnostics, and analysis-ready views. It is designed for practical engineering demos
          and iterative backend integration.
        </p>
        <div className="hero-actions">
          <Link className="btn btn-primary" to="/workbench">
            Go to Workbench
          </Link>
          <a
            className="btn btn-secondary"
            href="https://github.com/bstellato/mlopt"
            target="_blank"
            rel="noreferrer"
          >
            Reference Project
          </a>
        </div>
      </section>

      <section className="grid-three">
        {highlights.map((item, idx) => (
          <article className="card stagger-item" key={item.title} style={{ '--delay': `${idx * 80}ms` }}>
            <h2>{item.title}</h2>
            <p>{item.description}</p>
          </article>
        ))}
      </section>
    </div>
  )
}

export default HomePage
