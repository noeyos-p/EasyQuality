import { useState, useRef, useEffect } from 'react'
import ChatMessageComponent from './ChatMessage'
import InputArea from './InputArea'
import type { ChatMessage } from '../../types'
import { API_URL } from '../../types'

interface ChatPanelProps {
  isVisible: boolean
  onDocumentSelect: (docId: string, docType?: string, clause?: string) => void
}

export default function ChatPanel({ isVisible, onDocumentSelect }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)

  const [docNames, setDocNames] = useState<{ id: number; name: string }[]>([])
  const [suggestions, setSuggestions] = useState<{ id: number; name: string }[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [suggestionIndex, setSuggestionIndex] = useState(0)
  const [mentionTriggerPos, setMentionTriggerPos] = useState<number | null>(null)
  const [selectedDocs, setSelectedDocs] = useState<string[]>([])

  const [notifications, setNotifications] = useState<{ id: string; message: string }[]>([])
  const activeIntervals = useRef<Record<string, ReturnType<typeof setInterval>>>({})
  const notifiedIds = useRef<Set<string>>(new Set())

  useEffect(() => {
    return () => {
      Object.values(activeIntervals.current).forEach(clearInterval)
    }
  }, [])

  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set())

  const chatEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    const fetchDocNames = async () => {
      try {
        const response = await fetch(`${API_URL}/rag/doc-names`)
        const data = await response.json()
        if (data.doc_names) setDocNames(data.doc_names)
      } catch (error) {
        console.error('Failed to fetch doc names:', error)
      }
    }
    fetchDocNames()
  }, [])

  const updateMessage = (id: string, updates: Partial<ChatMessage>) => {
    setMessages(prev => prev.map(m => (m.id === id ? { ...m, ...updates } : m)))
  }

  const checkAllFinished = () => {
    setMessages(prev => {
      const stillWaiting = prev.some(m => m.isWaiting)
      if (!stillWaiting) setIsLoading(false)
      return prev
    })
  }

  const pollAnswer = (requestId: string, startTime: number, targetId: string) => {
    if (activeIntervals.current[targetId]) return

    let attempts = 0
    const maxAttempts = 180
    let lastPosition = -1

    const intervalId = setInterval(async () => {
      attempts += 1
      activeIntervals.current[targetId] = intervalId

      try {
        const token = localStorage.getItem('auth_token')
        const response = await fetch(`${API_URL}/chat/status/${requestId}`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        })
        const data = await response.json()

        if (data.status === 'completed') {
          clearInterval(intervalId)
          delete activeIntervals.current[targetId]

          const thinkingTime = Math.floor((Date.now() - startTime) / 1000)
          const result = data.result

          if (!sessionId && result?.session_id) {
            setSessionId(result.session_id)
          }

          updateMessage(targetId, {
            content: result?.answer || '답변을 생성하지 못했습니다.',
            timestamp: new Date(),
            thoughtProcess: result?.agent_log ? JSON.stringify(result.agent_log, null, 2) : 'Agent reasoning...',
            thinkingTime,
            evaluation_scores: result?.evaluation_scores,
            status: 'completed',
            ticket: data.ticket, // 티켓 번호 유지
            isWaiting: false,
          })

          checkAllFinished()

          if (!notifiedIds.current.has(targetId)) {
            notifiedIds.current.add(targetId)
            const notificationId = Math.random().toString(36).slice(2, 11)
            setNotifications(prev => [...prev, { id: notificationId, message: '에이전트의 답변이 도착했습니다!' }])
            setTimeout(() => {
              setNotifications(prev => prev.filter(n => n.id !== notificationId))
            }, 4000)
          }
        } else if (data.status === 'waiting') {
          if (data.position !== lastPosition || data.ticket) {
            lastPosition = data.position
            updateMessage(targetId, {
              status: 'waiting',
              queuePosition: data.position || 0,
              ticket: data.ticket
            })
          }
        } else if (data.status === 'processing') {
          updateMessage(targetId, {
            status: 'processing',
            queuePosition: 0,
          })
        } else if (data.status === 'error') {
          clearInterval(intervalId)
          delete activeIntervals.current[targetId]
          updateMessage(targetId, {
            content: `처리에 실패했습니다: ${data.error}`,
            status: 'error',
            isWaiting: false,
          })
          checkAllFinished()
        } else if (attempts >= maxAttempts) {
          clearInterval(intervalId)
          delete activeIntervals.current[targetId]
          updateMessage(targetId, {
            content: '답변 생성 시간이 초과되었습니다.',
            status: 'error',
            isWaiting: false,
          })
          checkAllFinished()
        }
      } catch (error) {
        console.error('Polling error:', error)
      }
    }, 2000)
  }

  const sendMessage = async () => {
    if (!inputMessage.trim()) return

    const currentInput = inputMessage
    const currentDocs = [...selectedDocs]
    const formattedContent = currentDocs.length > 0
      ? `${currentDocs.map(d => `@${d}`).join(' ')} ${currentInput}`
      : currentInput

    const assistantId = Math.random().toString(36).slice(2, 11)

    setMessages(prev => [
      ...prev,
      { role: 'user', content: formattedContent, timestamp: new Date() },
      {
        id: assistantId,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        status: 'waiting',
        isWaiting: true,
        queuePosition: 0,
      },
    ])

    setInputMessage('')
    setIsLoading(true)

    const startTime = Date.now()

    try {
      const token = localStorage.getItem('auth_token')
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          message: currentDocs.length > 0
            ? `[Selected Documents: ${currentDocs.join(', ')}]\n${currentInput}`
            : currentInput,
          session_id: sessionId,
          llm_model: 'gpt-4o-mini',
        }),
      })

      if (response.ok) {
        const data = await response.json()
        const requestId = data.request_id
        if (!sessionId && data.session_id) {
          setSessionId(data.session_id)
        }

        if (data.position || data.ticket) {
          updateMessage(assistantId, {
            queuePosition: data.position || 0,
            ticket: data.ticket,
            status: 'waiting'
          })
        }

        pollAnswer(requestId, startTime, assistantId)
      } else {
        const error = await response.json()
        updateMessage(assistantId, {
          content: `오류가 발생했습니다: ${error.detail}`,
          isWaiting: false,
          status: 'error',
        })
        checkAllFinished()
      }
    } catch (error) {
      updateMessage(assistantId, {
        content: `네트워크 오류: ${error}`,
        isWaiting: false,
        status: 'error',
      })
      checkAllFinished()
    } finally {
      setSelectedDocs([])
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    const isComposing = (e.nativeEvent as KeyboardEvent).isComposing || e.key === 'Process'
    if (isComposing) return

    // 입력창이 비어있을 때 Backspace로 마지막 선택 문서 태그 제거
    if (e.key === 'Backspace' && !inputMessage.trim() && selectedDocs.length > 0) {
      e.preventDefault()
      setSelectedDocs(prev => prev.slice(0, -1))
      return
    }

    if (e.key === 'Enter' && !e.shiftKey) {
      if (showSuggestions && suggestions.length > 0) {
        e.preventDefault()
        selectSuggestion(suggestions[suggestionIndex].name)
      } else {
        e.preventDefault()
        sendMessage()
      }
    } else if (showSuggestions) {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setSuggestionIndex(prev => (prev + 1) % suggestions.length)
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        setSuggestionIndex(prev => (prev - 1 + suggestions.length) % suggestions.length)
      } else if (e.key === 'Escape') {
        setShowSuggestions(false)
      }
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value
    const cursorPos = e.target.selectionStart
    setInputMessage(value)

    const lastAtPos = value.lastIndexOf('@', cursorPos - 1)
    if (lastAtPos !== -1) {
      const textAfterAt = value.substring(lastAtPos + 1, cursorPos)
      if (!textAfterAt.includes(' ')) {
        const filtered = docNames.filter(doc => doc.name.toLowerCase().includes(textAfterAt.toLowerCase()))
        setSuggestions(filtered)
        setShowSuggestions(filtered.length > 0)
        setSuggestionIndex(0)
        setMentionTriggerPos(lastAtPos)
        return
      }
    }
    setShowSuggestions(false)
  }

  const selectSuggestion = (name: string) => {
    if (mentionTriggerPos !== null) {
      const before = inputMessage.substring(0, mentionTriggerPos)
      const input = document.querySelector('.agent-input') as HTMLTextAreaElement
      const currentPos = input?.selectionStart || mentionTriggerPos + 1
      const afterAt = inputMessage.substring(currentPos)

      const newValue = before + (afterAt.startsWith(' ') ? afterAt.substring(1) : afterAt)
      setInputMessage(newValue)

      if (!selectedDocs.includes(name)) {
        setSelectedDocs(prev => [...prev, name])
      }
      setShowSuggestions(false)
      setMentionTriggerPos(null)
    }
  }

  const removeSelectedDoc = (docId: string) => {
    setSelectedDocs(prev => prev.filter(id => id !== docId))
  }

  const toggleSection = (section: string) => {
    const newSet = new Set(expandedSections)
    if (newSet.has(section)) newSet.delete(section)
    else newSet.add(section)
    setExpandedSections(newSet)
  }

  return (
    <aside className={`flex-shrink-0 bg-dark-deeper border-l border-dark-border flex flex-col overflow-hidden transition-[width,opacity,border-color] duration-300 ease-in-out ${!isVisible ? 'w-0 opacity-0 border-l-transparent pointer-events-none' : 'w-[420px]'}`}>
      <div className="flex justify-between items-center px-4 py-2 h-[35px] border-b border-dark-border">
        <span className="text-[13px] font-medium text-txt-primary">Agent Chat</span>
      </div>

      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
          {messages.map((msg, index) => (
            <div key={index} className="flex flex-col gap-2">
              <ChatMessageComponent
                msg={msg}
                index={index}
                expandedSections={expandedSections}
                toggleSection={toggleSection}
                onDocumentSelect={onDocumentSelect}
              />
            </div>
          ))}

          <div ref={chatEndRef} />
        </div>

        <InputArea
          inputMessage={inputMessage}
          isLoading={isLoading}
          selectedDocs={selectedDocs}
          suggestions={suggestions}
          showSuggestions={showSuggestions}
          suggestionIndex={suggestionIndex}
          onInputChange={handleInputChange}
          onKeyDown={handleKeyPress}
          onSend={sendMessage}
          onSelectSuggestion={selectSuggestion}
          onRemoveDoc={removeSelectedDoc}
        />
      </div>

      <div className="fixed bottom-24 right-6 flex flex-col gap-2 z-[3000]">
        {notifications.map(n => (
          <div key={n.id} className="bg-accent-blue text-white px-4 py-3 rounded-lg shadow-2xl flex items-center gap-3 animate-slide-in-right">
            <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
            <span className="text-[13px] font-medium">{n.message}</span>
            <button
              onClick={() => setNotifications(prev => prev.filter(notif => notif.id !== n.id))}
              className="ml-2 hover:opacity-70"
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </aside>
  )
}
