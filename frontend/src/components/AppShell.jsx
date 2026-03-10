import { NavLink } from 'react-router-dom'

const navItems = [
  { to: '/', label: 'Home' },
  { to: '/workbench', label: 'Workbench' },
]

function AppShell({ children }) {
  return (
    <>
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>

      <div className="bg-orb bg-orb-one" aria-hidden="true" />
      <div className="bg-orb bg-orb-two" aria-hidden="true" />

      <header className="site-header">
        <div className="shell shell-header">
          <NavLink className="brand" to="/">
            <span className="brand-mark" aria-hidden="true">
              M
            </span>
            <span className="brand-text">
              <strong>mlopt</strong>
              <small>Frontend MVP</small>
            </span>
          </NavLink>

          <nav aria-label="Primary navigation">
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
          <p>Built with React + Vite. Existing backend logic remains unchanged.</p>
        </div>
      </footer>
    </>
  )
}

export default AppShell
