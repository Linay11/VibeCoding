import { NavLink } from 'react-router-dom'

const navItems = [
  { to: '/', label: '总览' },
  { to: '/workbench', label: '实验台' },
]

function AppShell({ children }) {
  return (
    <>
      <a className="skip-link" href="#main-content">
        跳转到主内容
      </a>

      <div className="bg-orb bg-orb-one" aria-hidden="true" />
      <div className="bg-orb bg-orb-two" aria-hidden="true" />

      <header className="site-header">
        <div className="shell shell-header">
          <NavLink className="brand" to="/">
            <span className="brand-mark" aria-hidden="true">
              核
            </span>
            <span className="brand-text">
              <strong>优化实验控制台</strong>
              <small>实验运行与结果分析</small>
            </span>
          </NavLink>

          <nav aria-label="主导航">
            <ul className="nav-list">
              {navItems.map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    className={({ isActive }) =>
                      isActive ? 'nav-link nav-link-active' : 'nav-link'
                    }
                  >
                    {item.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      </header>

      <main id="main-content" className="site-main shell">
        {children}
      </main>

      <footer className="site-footer">
        <div className="shell footer-content">
          <p>由 React + Vite 构建的优化实验前端。仅升级界面与交互表达，核心求解逻辑保持不变。</p>
        </div>
      </footer>
    </>
  )
}

export default AppShell
