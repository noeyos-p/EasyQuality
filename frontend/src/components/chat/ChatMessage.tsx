import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import MermaidRenderer from '../graph/MermaidRenderer'
import type { ChatMessage as ChatMessageType } from '../../types'
import { SCORE_COLORS } from '../../types'

interface ChatMessageProps {
  msg: ChatMessageType
  index: number
  expandedSections: Set<string>
  toggleSection: (section: string) => void
  onDocumentSelect: (docId: string) => void
}

const DOC_PATTERN = /(EQ-(?:SOP|WI)-\d{5}(?:\([\d.,\s]+\))?)/g

function processText(text: string, onDocumentSelect: (docId: string) => void) {
  const parts = text.split(DOC_PATTERN)
  return parts.map((part, i) => {
    if (DOC_PATTERN.test(part)) {
      const docId = part.split('(')[0].replace(/^@/, '')
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
    return part
  })
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function recurse(node: any, onDocumentSelect: (docId: string) => void): any {
  if (typeof node === 'string') return processText(node, onDocumentSelect)
  if (Array.isArray(node)) return node.map(n => recurse(n, onDocumentSelect))
  if (node?.props?.children) {
    return { ...node, props: { ...node.props, children: recurse(node.props.children, onDocumentSelect) } }
  }
  return node
}

export default function ChatMessage({ msg, index, expandedSections, toggleSection, onDocumentSelect }: ChatMessageProps) {
  if (msg.role === 'user') {
    return (
      <div className="bg-dark-light rounded-lg p-3 border border-dark-border">
        <div className="flex-1 text-[13px] text-txt-primary">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              p({ children }) {
                return <p>{recurse(children, onDocumentSelect)}</p>
              }
            }}
          >
            {msg.content}
          </ReactMarkdown>
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
          components={{
            p({ children }) {
              return <p>{recurse(children, onDocumentSelect)}</p>
            },
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
