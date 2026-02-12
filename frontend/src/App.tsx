import { useState, useEffect } from 'react'
import Sidebar from './components/layout/Sidebar'
import DocumentManagementPanel from './components/document/DocumentManagementPanel'
import ChangeHistoryPanel from './components/history/ChangeHistoryPanel'
import GraphVisualization from './components/graph/GraphVisualization'
import DocumentViewer from './components/document/DocumentViewer'
import ChatPanel from './components/chat/ChatPanel'
import VersionDiffViewer from './components/history/VersionDiffViewer'
import { API_URL } from './types'

function App() {
  // 서버 상태
  const [isConnected, setIsConnected] = useState(false)
  const [agentStatus, setAgentStatus] = useState<string>('연결 확인 중...')

  // 문서 상태
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null)
  const [documentContent, setDocumentContent] = useState<string | null>(null)
  const [isEditing, setIsEditing] = useState(false)
  const [editedContent, setEditedContent] = useState<string>('')
  const [isSaving, setIsSaving] = useState(false)

  // UI 상태
  const [activePanel, setActivePanel] = useState<'documents' | 'visualization' | 'history' | null>(null)
  const [isLeftVisible, setIsLeftVisible] = useState(true)
  const [isRightVisible, setIsRightVisible] = useState(true)
  const [isDraggingOver, setIsDraggingOver] = useState(false)

  // 비교 모드 상태
  const [isComparing, setIsComparing] = useState(false)
  const [diffData, setDiffData] = useState<any>(null)

  // 🆕 전역 알림(Toast) 상태
  const [toasts, setToasts] = useState<{ id: string, message: string, type: 'success' | 'error' | 'info' }[]>([])

  const addToast = (message: string, type: 'success' | 'error' | 'info' = 'success') => {
    const id = Math.random().toString(36).substr(2, 9)
    setToasts(prev => [...prev, { id, message, type }])
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id))
    }, 5000)
  }

  // 백엔드 연결 확인
  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        const res = await fetch(`${API_URL}/health`)
        if (res.ok) {
          setIsConnected(true)
          setAgentStatus('Agent Ready')
        } else {
          setIsConnected(false)
          setAgentStatus('Connection Failed')
        }
      } catch {
        setIsConnected(false)
        setAgentStatus('Server Offline')
      }
    }
    checkBackendStatus()
  }, [])

  const handleDocumentSelect = async (docId: string, content?: string) => {
    setSelectedDocument(docId)
    if (content) {
      setDocumentContent(content)
      setEditedContent(content)
      setIsEditing(false)
    } else {
      try {
        const response = await fetch(`${API_URL}/rag/document/${docId}/content`)
        const data = await response.json()
        if (data.content) {
          setDocumentContent(data.content)
          setEditedContent(data.content)
        } else {
          setDocumentContent('내용을 불러올 수 없습니다.')
          setEditedContent('내용을 불러올 수 없습니다.')
        }
        setIsEditing(false)
      } catch {
        setDocumentContent('문서 내용을 가져오는 중 오류가 발생했습니다.')
        setEditedContent('문서 내용을 가져오는 중 오류가 발생했습니다.')
        setIsEditing(false)
      }
    }
  }

  const handleSaveDocument = async () => {
    if (!selectedDocument) return
    setIsSaving(true) // 버튼 비활성화용
    try {
      const response = await fetch(`${API_URL}/rag/document/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_name: selectedDocument, content: editedContent }),
      })

      const data = await response.json()

      if (response.ok) {
        addToast(`'${selectedDocument}' 수정 요청이 접수되었습니다. 배경 분석 중...`, 'info')
        setDocumentContent(editedContent)
        setIsEditing(false)

        // 🆕 비동기 완료 감지를 위한 이벤트 발송 (DocumentManagementPanel 등에서 수신 가능)
        window.dispatchEvent(new CustomEvent('document_processing_start', {
          detail: { docName: selectedDocument, type: 'save' }
        }))

      } else {
        addToast(`저장 요청 실패: ${data.detail || '알 수 없는 오류'}`, 'error')
      }
    } catch (error) {
      console.error('저장 에러:', error)
      addToast('저장 중 연결 오류가 발생했습니다.', 'error')
    } finally {
      setIsSaving(false)
    }
  }

  const handleCompare = async (docName: string, v1: string, v2: string) => {
    try {
      const response = await fetch(`${API_URL}/rag/document/${encodeURIComponent(docName)}/diff?v1=${v1}&v2=${v2}`)
      if (response.ok) {
        const data = await response.json()
        setDiffData(data)
        setIsComparing(true)
        setActivePanel(null) // 사이드 패널 닫기 (공간 확보)
      } else {
        const error = await response.json()
        alert(`비교 실패: ${error.detail || '알 수 없는 오류'}`)
      }
    } catch (error) {
      console.error('비교 요청 오류:', error)
      alert('비교 요청 중 오류가 발생했습니다.')
    }
  }

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* 헤더 */}
      <header className="flex justify-between items-center h-[35px] bg-dark-deeper border-b border-dark-border px-4">
        <div className="flex items-center gap-3">
          <button
            className={`border-none py-1 px-2 text-[14px] rounded cursor-pointer flex items-center justify-center transition-all duration-200 ${isLeftVisible ? 'bg-transparent text-txt-secondary hover:bg-dark-hover hover:text-accent' : 'bg-accent/10 text-accent'}`}
            onClick={() => setIsLeftVisible(!isLeftVisible)}
            title={isLeftVisible ? '사이드바 접기' : '사이드바 펴기'}
          >
            {isLeftVisible ? '◀' : '▶'}
          </button>
          <span className="text-[13px] text-txt-primary">Orchestrator Agent</span>

          {selectedDocument && (
            <div className="flex gap-2 ml-4">
              {!isEditing ? (
                <button
                  className="bg-dark-hover border border-dark-border text-accent py-1 px-3 text-[11px] rounded cursor-pointer transition-all duration-200 hover:bg-dark-border hover:border-txt-secondary"
                  onClick={() => setIsEditing(true)}
                >
                  수정
                </button>
              ) : (
                <>
                  <button
                    className="bg-dark-hover border border-dark-border text-[#f48fb1] py-1 px-3 text-[11px] rounded cursor-pointer transition-all duration-200 hover:bg-dark-border hover:border-txt-secondary"
                    onClick={() => { setIsEditing(false); setEditedContent(documentContent || '') }}
                  >
                    취소
                  </button>
                  <button
                    className="bg-accent-blue text-white border-accent-blue py-1 px-3 text-[11px] rounded cursor-pointer transition-all duration-200 hover:bg-[#0062a3]"
                    onClick={handleSaveDocument}
                  >
                    저장
                  </button>
                </>
              )}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          <span className={`text-[12px] ${isConnected ? 'text-accent' : 'text-[#f48771]'}`}>
            {isConnected ? '[OK]' : '[ERROR]'} {agentStatus}
          </span>
          <button
            className={`border-none py-1 px-2 text-[14px] rounded cursor-pointer flex items-center justify-center transition-all duration-200 ${isRightVisible ? 'bg-transparent text-txt-secondary hover:bg-dark-hover hover:text-accent' : 'bg-accent/10 text-accent'}`}
            onClick={() => setIsRightVisible(!isRightVisible)}
            title={isRightVisible ? '채팅 패널 접기' : '채팅 패널 펴기'}
          >
            {isRightVisible ? '▶' : '◀'}
          </button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* 사이드바 아이콘 */}
        <Sidebar activePanel={activePanel} onPanelChange={(panel) => {
          setActivePanel(panel)
          if (panel) setIsLeftVisible(true)
        }} />

        {/* 사이드 패널 */}
        <div className={`flex-shrink-0 bg-dark-deeper border-r border-dark-border flex flex-col overflow-hidden transition-[width,opacity,border-color] duration-300 ease-in-out ${!isLeftVisible || !activePanel || activePanel === 'visualization' ? 'w-0 opacity-0 border-r-transparent pointer-events-none' : 'w-80'}`}>
          {activePanel === 'documents' && (
            <DocumentManagementPanel
              onDocumentSelect={handleDocumentSelect}
              onNotify={addToast}
            />
          )}
          {activePanel === 'history' && (
            <ChangeHistoryPanel onCompare={handleCompare} selectedDocName={selectedDocument} />
          )}
        </div>

        {/* 가운데: 문서 뷰어 또는 그래프 */}
        <main
          className={`flex-1 bg-dark-bg overflow-y-auto flex flex-col transition-all duration-300 relative ${isDraggingOver ? 'outline outline-2 outline-accent-blue outline-offset-[-2px]' : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            e.dataTransfer.dropEffect = 'copy'
            if (!isDraggingOver) setIsDraggingOver(true)
          }}
          onDragLeave={() => setIsDraggingOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setIsDraggingOver(false)
            const docId = e.dataTransfer.getData('text/plain')
            if (docId) handleDocumentSelect(docId)
          }}
        >
          {isDraggingOver && (
            <div className="absolute inset-0 bg-accent-blue/10 flex items-center justify-center z-50 pointer-events-none">
              <div className="flex flex-col items-center gap-3 text-txt-primary">
                <span className="text-[48px]">📄</span>
                <span className="text-[16px]">여기에 드롭하여 문서 열기</span>
              </div>
            </div>
          )}

          {activePanel === 'visualization' ? (
            <GraphVisualization
              onNodeClick={(docId) => handleDocumentSelect(docId)}
              onSwitchToDocuments={() => setActivePanel('documents')}
            />
          ) : isComparing && diffData ? (
            <VersionDiffViewer
              diffData={diffData}
              onClose={() => {
                setIsComparing(false)
                setDiffData(null)
              }}
            />
          ) : selectedDocument && documentContent ? (
            <DocumentViewer
              selectedDocument={selectedDocument}
              documentContent={documentContent}
              isEditing={isEditing}
              editedContent={editedContent}
              setEditedContent={setEditedContent}
            />
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-txt-secondary">
              <div className="text-[64px] mb-4 opacity-50">[FILE]</div>
              <h2 className="text-[18px] font-medium mb-2 text-txt-primary">Select a document</h2>
            </div>
          )}
        </main>

        {/* 채팅 패널 */}
        <ChatPanel
          isVisible={isRightVisible}
          onDocumentSelect={handleDocumentSelect}
        />
      </div>

      {/* 🆕 품격 있는 Toast 알림 */}
      <div className="fixed top-12 right-6 z-[3000] flex flex-col gap-3 pointer-events-none">
        {toasts.map(toast => (
          <div
            key={toast.id}
            className={`pointer-events-auto min-w-[300px] p-4 rounded-lg shadow-2xl border flex items-center gap-3 animate-slide-in-right
              ${toast.type === 'success' ? 'bg-[#1e1e1e] border-[#4ec9b0] text-[#4ec9b0]' :
                toast.type === 'error' ? 'bg-[#1e1e1e] border-[#f48771] text-[#f48771]' :
                  'bg-[#1e1e1e] border-accent-blue text-accent-blue'}`}
          >
            <span className="text-[18px]">
              {toast.type === 'success' ? '✓' : toast.type === 'error' ? '⚠' : 'ℹ'}
            </span>
            <div className="flex-1">
              <p className="m-0 text-[13px] font-medium leading-normal">{toast.message}</p>
            </div>
          </div>
        ))}
      </div>

      <style>{`
        @keyframes slide-in-right {
          from { transform: translateX(100%); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }
        .animate-slide-in-right {
          animation: slide-in-right 0.3s ease-out forwards;
        }
      `}</style>
    </div>
  )
}

export default App
