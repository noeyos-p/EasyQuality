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
    setIsSaving(true)
    try {
      const response = await fetch(`${API_URL}/rag/document/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_name: selectedDocument, content: editedContent }),
      })
      if (response.ok) {
        const data = await response.json()
        setDocumentContent(editedContent)
        setIsEditing(false)
        alert(`문서가 저장되었습니다. (새 버전: ${data.version})`)
      } else {
        const errorData = await response.json()
        alert(`저장 실패: ${errorData.detail || '알 수 없는 오류'}`)
      }
    } catch {
      alert('저장 중 오류가 발생했습니다.')
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
            <DocumentManagementPanel onDocumentSelect={handleDocumentSelect} />
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

      {/* 저장 중 오버레이 */}
      {isSaving && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-[2000]">
          <div className="bg-[#2d2d2d] border border-dark-border rounded-lg p-8 flex flex-col items-center gap-4 text-center">
            <div className="w-10 h-10 border-4 border-dark-border border-t-accent-blue rounded-full animate-spin"></div>
            <p className="text-txt-primary text-[14px] m-0">문서를 분석하고 저장하는 중입니다...</p>
            <span className="text-txt-secondary text-[12px]">이 작업은 최대 1분 정도 소요될 수 있습니다.</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
