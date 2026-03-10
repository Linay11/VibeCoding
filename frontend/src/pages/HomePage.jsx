import { Link } from 'react-router-dom'

const highlights = [
  {
    title: 'Model-Aware Workflow',
    description:
      'Move from scenario selection to run output and strategy comparison without switching tools.',
  },
  {
    title: 'Backend-Safe Integration',
    description:
      'The UI reads API data when available and never changes existing optimization core logic.',
  },
  {
    title: 'Practical MVP Defaults',
    description:
      'If backend endpoints are offline, the app falls back to demo data so pages stay usable.',
  },
]

function HomePage() {
  return (
    <div className="fade-in">
      <section className="hero card">
        <p className="eyebrow">Machine Learning Optimizer</p>
        <h1>Operate optimization experiments with a modern UI.</h1>
        <p className="lead">
          This frontend MVP adds a clean shell around your optimizer workflows: clear navigation,
          guided run actions, and readable run summaries. It is built for progressive backend
          integration.
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
