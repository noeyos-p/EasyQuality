import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import MermaidRenderer from '../graph/MermaidRenderer'
import type { ChatMessage as ChatMessageType } from '../../types'
import { SCORE_COLORS } from '../../types'

interface ChatMessageProps {
  msg: ChatMessageType
  index: number
  expandedSections: Set<string>
  toggleSection: (section: string) => void
  onDocumentSelect: (docId: string, docType?: string, version?: string, clause?: string) => void
}

const DOC_PATTERN = /(EQ-(?:SOP|WI|FRM)-\d{4,6}(?:\([\d.,\s]+\))?)/g
const DOC_TOKEN_PATTERN = /^EQ-(?:SOP|WI|FRM)-\d{4,6}(?:\([\d.,\s]+\))?$/

function processText(text: string, onDocumentSelect: (docId: string, docType?: string, version?: string, clause?: string) => void) {
  const parts = text.split(DOC_PATTERN)
  return parts.map((part, i) => {
    if (DOC_TOKEN_PATTERN.test(part)) {
      const token = part
        .replace(/^@/, '')
        .replace(/[.,;:!?]+$/, '')
        .trim()
      const match = token.match(/^(EQ-(?:SOP|WI|FRM)-\d{4,6})(?:\(([^)]+)\))?$/)
      const docId = match?.[1] ?? token
      const rawClause = match?.[2]?.trim()
      const clauses = rawClause
        ? rawClause.split(',').map(v => v.trim()).filter(Boolean)
        : []

      if (clauses.length === 0) {
        return (
          <span
            key={i}
            className="text-accent underline cursor-pointer font-medium px-1 py-[1px] rounded transition-all duration-200 hover:bg-white/10 hover:text-accent-hover"
            onClick={() => onDocumentSelect(docId)}
          >
            {part}
          </span>
        )
      }

      return (
        <span key={i} className="px-1 py-[1px]">
          <span
            className="text-accent underline cursor-pointer font-medium rounded transition-all duration-200 hover:bg-white/10 hover:text-accent-hover"
            onClick={() => onDocumentSelect(docId)}
          >
            {docId}
          </span>
          <span>(</span>
          {clauses.map((clause, idx) => (
            <span key={`${docId}-${clause}-${idx}`}>
              <span
                className="text-accent underline cursor-pointer font-medium rounded transition-all duration-200 hover:bg-white/10 hover:text-accent-hover"
                onClick={() => onDocumentSelect(docId, undefined, undefined, clause)}
              >
                {clause}
              </span>
              {idx < clauses.length - 1 ? <span>, </span> : null}
            </span>
          ))}
          <span>)</span>
        </span>
      )
    }
    return part
  })
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function recurse(node: any, onDocumentSelect: (docId: string, docType?: string, version?: string, clause?: string) => void): any {
  if (typeof node === 'string') return processText(node, onDocumentSelect)
  if (Array.isArray(node)) return node.map(n => recurse(n, onDocumentSelect))
  if (node?.props?.children) {
    return { ...node, props: { ...node.props, children: recurse(node.props.children, onDocumentSelect) } }
  }
  return node
}

export default function ChatMessage({ msg, index, expandedSections, toggleSection, onDocumentSelect }: ChatMessageProps) {
  const markdownTextComponents = {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    p({ children }: any) {
      return <p>{recurse(children, onDocumentSelect)}</p>
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    li({ children }: any) {
      return <li>{recurse(children, onDocumentSelect)}</li>
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    h1({ children }: any) {
      return <h1>{recurse(children, onDocumentSelect)}</h1>
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    h2({ children }: any) {
      return <h2>{recurse(children, onDocumentSelect)}</h2>
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    h3({ children }: any) {
      return <h3>{recurse(children, onDocumentSelect)}</h3>
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    h4({ children }: any) {
      return <h4>{recurse(children, onDocumentSelect)}</h4>
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    h5({ children }: any) {
      return <h5>{recurse(children, onDocumentSelect)}</h5>
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    h6({ children }: any) {
      return <h6>{recurse(children, onDocumentSelect)}</h6>
    },
  }

  if (msg.role === 'user') {
    return (
      <div className="bg-dark-light rounded-lg p-3 border border-dark-border">
        <div className="flex-1 text-[13px] text-txt-primary">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeRaw]}
            components={markdownTextComponents}
          >
            {msg.content}
          </ReactMarkdown>
        </div>
      </div>
    )
  }

  if (msg.status === 'waiting' || msg.status === 'processing' || msg.isWaiting) {
    const isProcessing = msg.status === 'processing'
    return (
      <div className="flex flex-col gap-2">
        <div className="bg-dark-light rounded-lg p-3 border border-dark-border animate-pulse">
          <div className="flex items-center gap-3">
            <div className={`w-3 h-3 border-2 ${isProcessing ? 'border-accent' : 'border-accent-blue'} border-t-transparent rounded-full animate-spin`} />
            <span className="text-[13px] text-txt-secondary">
              {isProcessing
                ? '에이전트가 답변을 작성하고 있습니다...'
                : `답변 대기 중... (${msg.queuePosition ? `순번 ${msg.queuePosition}번째` : '순번 계산 중...'})`}
            </span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      {/* Thought Process */}
      {msg.thoughtProcess && (
        <div className="bg-dark-light rounded overflow-hidden border border-dark-border">
          <div
            className="flex items-center gap-2 px-3 py-2 cursor-pointer transition-colors duration-200 select-none hover:bg-dark-hover"
            onClick={() => toggleSection(`thought-${index}`)}
          >
            <span className="text-[10px] text-txt-muted w-3">
              {expandedSections.has(`thought-${index}`) ? '▼' : '▶'}
            </span>
            <span className="text-[13px] text-txt-secondary font-medium">Show Reasoning</span>
          </div>
          {expandedSections.has(`thought-${index}`) && (
            <pre className="p-3 border-t border-dark-border text-[13px] text-txt-primary leading-[1.6] bg-dark-deeper">
              {msg.thoughtProcess}
            </pre>
          )}
        </div>
      )}

      {/* 답변 본문 */}
      <div className="response-body">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeRaw]}
          components={{
            ...markdownTextComponents,
            code({ node, inline, className, children, ...props }: any) {
              const match = /language-(\w+)/.exec(className || '')
              const language = match ? match[1] : ''
              if (!inline && language === 'mermaid') {
                return <MermaidRenderer chart={String(children).replace(/\n$/, '')} />
              }
              return !inline ? (
                <pre className={className}><code {...props}>{children}</code></pre>
              ) : (
                <code className={className} {...props}>{children}</code>
              )
            }
          }}
        >
          {msg.content}
        </ReactMarkdown>
      </div>

      {msg.thinkingTime && (
        <div className="text-[11px] text-txt-muted mt-2">Time: {msg.thinkingTime}s</div>
      )}

      {/* 평가 점수 */}
      {msg.evaluation_scores && (
        <div className="mt-3 border-t border-dark-border pt-2">
          <div
            className="flex items-center gap-2 p-2 cursor-pointer rounded transition-colors duration-200 hover:bg-dark-hover"
            onClick={() => toggleSection(`eval-${index}`)}
          >
            <span className="text-[10px] text-txt-muted w-3">
              {expandedSections.has(`eval-${index}`) ? '▼' : '▶'}
            </span>
            <span className="text-[13px] font-semibold text-txt-secondary">
              🔍 평가 점수
              {msg.evaluation_scores.average_score && (
                <span className="text-[12px] text-accent font-bold ml-2">
                  ({msg.evaluation_scores.average_score.toFixed(1)}/5.0)
                </span>
              )}
            </span>
          </div>

          {expandedSections.has(`eval-${index}`) && (
            <div className="p-3 bg-dark-deeper rounded mt-2">
              {(['faithfulness', 'groundness', 'relevancy', 'correctness'] as const).map((key) => {
                const score = msg.evaluation_scores![key]
                if (!score) return null
                const labels: Record<string, string> = {
                  faithfulness: '충실성 (Faithfulness)',
                  groundness: '근거성 (Groundness)',
                  relevancy: '관련성 (Relevancy)',
                  correctness: '정확성 (Correctness)',
                }
                return (
                  <div key={key} className="mb-4 pb-3 border-b border-dark-border last:mb-0 last:pb-0 last:border-b-0">
                    <span className="text-[12px] font-semibold text-txt-primary mr-2">{labels[key]}:</span>
                    <span className={`text-sm font-bold py-0.5 px-2 rounded ml-2 ${SCORE_COLORS[score.score] ?? ''}`}>
                      {score.score}/5
                    </span>
                    <div className="text-[11px] text-txt-secondary mt-1.5 leading-[1.4] pl-1 border-l-2 border-dark-border">
                      {score.reasoning}
                    </div>
                    {score.rdb_verification && (
                      <div className="mt-2.5 p-2.5 bg-dark-deeper rounded border border-dark-border">
                        <div className="text-[11px] font-bold text-accent mb-2">📊 RDB 검증 결과</div>
                        <div className="flex gap-4 mb-2">
                          <span className="text-[11px] text-txt-secondary">
                            정확도: <strong className="text-accent text-[13px]">{score.rdb_verification.accuracy_rate}%</strong>
                          </span>
                          <span className="text-[11px] text-txt-secondary">
                            검증됨: {score.rdb_verification.verified_citations}/{score.rdb_verification.total_citations}
                          </span>
                        </div>
                        {score.rdb_verification.incorrect_citations.length > 0 && (
                          <div>
                            <strong>⚠️ 틀린 인용:</strong>
                            <ul>
                              {score.rdb_verification.incorrect_citations.map((citation, i) => (
                                <li key={i}>{citation}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                        <details>
                          <summary className="text-[11px] text-txt-secondary cursor-pointer">상세 검증 결과</summary>
                          <pre className="text-[11px]">{score.rdb_verification.verification_details}</pre>
                        </details>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
