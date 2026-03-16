import { Link } from 'react-router-dom'

const highlights = [
  {
    titleCn: '实验优先的工作流',
    titleEn: 'Workflow First',
    description:
      '从场景选择到运行结果、趋势分析与策略解释，在同一页面完成闭环。',
  },
  {
    titleCn: '运行模式透明可见',
    titleEn: 'Mode Transparency',
    description:
      '一眼区分真实后端执行、兼容模式与前端回退，不再“黑箱”。',
  },
  {
    titleCn: '稳健的演示体验',
    titleEn: 'Resilient Experience',
    description:
      '即使后端暂不可达，界面仍可交互并明确提示当前状态。',
  },
]

function HomePage() {
  return (
    <div className="fade-in home-overview">
      <section className="hero card overview-hero">
        <div className="overview-hero-grid">
          <div className="overview-hero-main">
            <p className="eyebrow">Optimization Lab · 优化实验仪表台</p>
            <p className="overview-kicker">Overview / 总览</p>
            <h1 className="overview-title-cn">
              <span className="overview-title-core">以更清晰的上下文，读懂每一次</span>
              <span className="overview-title-mark">
                <span className="overview-title-mark-text">优化运行</span>
                <span className="title-fragment title-fragment-1" aria-hidden="true" />
                <span className="title-fragment title-fragment-2" aria-hidden="true" />
                <span className="title-fragment title-fragment-3" aria-hidden="true" />
                <span className="title-fragment title-fragment-4" aria-hidden="true" />
                <span className="title-fragment title-fragment-5" aria-hidden="true" />
                <span className="title-fragment title-fragment-6" aria-hidden="true" />
              </span>
              。
            </h1>
            <p className="overview-title-en" aria-hidden="true">
              Constraint-Aware Optimization
            </p>
            <p className="lead">
              这个前端界面把你的优化流程组织成结构化摘要、稳定诊断与可分析视图。
              你可以更快定位问题、解释结果，并持续迭代后端能力。
            </p>
            <div className="hero-actions">
              <Link className="btn btn-primary" to="/workbench">
                进入实验台
              </Link>
              <a
                className="btn btn-secondary"
                href="https://github.com/bstellato/mlopt"
                target="_blank"
                rel="noreferrer"
              >
                参考项目
              </a>
            </div>
          </div>

          <aside className="overview-hero-aside">
            <div className="overview-aside-particles" aria-hidden="true">
              <span className="poem-particle poem-particle-1" />
              <span className="poem-particle poem-particle-2" />
              <span className="poem-particle poem-particle-3" />
              <span className="poem-particle poem-particle-4" />
            </div>
            <p className="overview-aside-title">OPERATIONS POEM</p>
            <p className="overview-aside-cn">在约束与决策之间，让每一次运行都可解释。</p>
            <p className="overview-aside-en">Shape constraints. Reveal decisions.</p>
            <p className="overview-aside-sign">Observe · Reduce · Solve</p>
          </aside>
        </div>

        <div className="overview-rhythm" aria-hidden="true">
          <span>Signal / 信号</span>
          <span>Constraint / 约束</span>
          <span>Decision / 决策</span>
        </div>
      </section>

      <section className="grid-three">
        {highlights.map((item, idx) => (
          <article className="card stagger-item overview-card" key={item.titleCn} style={{ '--delay': `${idx * 80}ms` }}>
            <p className="overview-card-en">{item.titleEn}</p>
            <h2>{item.titleCn}</h2>
            <p>{item.description}</p>
          </article>
        ))}
      </section>
    </div>
  )
}

export default HomePage
