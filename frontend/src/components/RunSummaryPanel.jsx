import { memo } from 'react'

function RunSummaryPanel({
  scenario,
  sourceLabel,
  runModeLabel,
  modeReason,
  generatedTime,
  sourceCode,
  modeCode,
}) {
  const reasonText = modeReason && modeReason.trim() ? modeReason : '当前无补充说明。'
  const generatedText = generatedTime && generatedTime.trim() ? generatedTime : '尚未生成'
  const sourceLabelText = String(sourceLabel || '加载中')
  const runModeLabelText = String(runModeLabel || '不可用')
  const sourceClass = sourceCode === 'fallback' ? 'fallback' : sourceCode === 'api' ? 'api' : 'loading'
  const modeClass = modeCode === 'real' ? 'mode-real' : modeCode === 'compat' ? 'mode-compat' : modeCode === 'fallback' ? 'mode-fallback' : 'mode-none'

  return (
    <section className="card run-summary" aria-label="运行摘要">
      <div className="summary-head">
        <h2>运行摘要</h2>
        <p>当前结果快照的关键上下文。</p>
      </div>

      <div className="summary-grid">
        <article className="summary-item">
          <p className="summary-label">场景</p>
          <p className="summary-value">{scenario || '未选择场景'}</p>
        </article>

        <article className="summary-item">
          <p className="summary-label">数据来源</p>
          <p className="summary-value">
            <span className={`summary-badge source-${sourceClass}`}>
              {sourceLabelText}
            </span>
          </p>
        </article>

        <article className="summary-item">
          <p className="summary-label">运行模式</p>
          <p className="summary-value">
            <span className={`summary-badge ${modeClass}`}>
              {runModeLabelText}
            </span>
          </p>
        </article>

        <article className="summary-item">
          <p className="summary-label">生成时间</p>
          <p className="summary-value">{generatedText}</p>
        </article>

        <article className="summary-item summary-item-wide">
          <p className="summary-label">模式说明</p>
          <p className="summary-value summary-reason">{reasonText}</p>
        </article>
      </div>
    </section>
  )
}

export default memo(RunSummaryPanel)
