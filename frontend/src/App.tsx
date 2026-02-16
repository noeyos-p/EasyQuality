import { useState, useEffect } from 'react'
import Sidebar from './components/layout/Sidebar'
import DocumentManagementPanel from './components/document/DocumentManagementPanel'
import ChangeHistoryPanel from './components/history/ChangeHistoryPanel'
import GraphVisualization from './components/graph/GraphVisualization'
import DocumentViewer from './components/document/DocumentViewer'
import ChatPanel from './components/chat/ChatPanel'
import VersionDiffViewer from './components/history/VersionDiffViewer'
import AuthModal from './components/auth/AuthModal'
import { useAuth } from './hooks/useAuth'
import { API_URL } from './types'
import TaskNotification from './components/layout/TaskNotification'
import Toast from './components/layout/Toast'
import type { TaskStatus } from './components/layout/TaskNotification'
import type { ToastMessage } from './components/layout/Toast'

function App() {
  // 인증 상태
  const { isAuthenticated, user, login, register, logout } = useAuth()

  // 서버 상태
  const [isConnected, setIsConnected] = useState(false)
  const [agentStatus, setAgentStatus] = useState<string>('연결 확인 중...')

  // 문서 상태
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null)
  const [selectedClause, setSelectedClause] = useState<string | null>(null)
  const [documentContent, setDocumentContent] = useState<string | null>(null)
  const [isEditing, setIsEditing] = useState(false)
  const [editedContent, setEditedContent] = useState('')

  // OnlyOffice 에디터 상태
  const [isOnlyOfficeMode, setIsOnlyOfficeMode] = useState(false)
  const [onlyOfficeEditorMode, setOnlyOfficeEditorMode] = useState<'view' | 'edit'>('view')
  const [onlyOfficeConfig, setOnlyOfficeConfig] = useState<object | null>(null)
  const [onlyOfficeServerUrl, setOnlyOfficeServerUrl] = useState<string>('')

  // UI 상태
  const [activePanel, setActivePanel] = useState<'documents' | 'visualization' | 'history' | null>(null)
  const [isLeftVisible, setIsLeftVisible] = useState(true)
  const [isRightVisible, setIsRightVisible] = useState(true)
  const [isDraggingOver, setIsDraggingOver] = useState(false)

  // 비교 모드 상태
  const [isComparing, setIsComparing] = useState(false)
  const [diffData, setDiffData] = useState<any>(null)

  // 🔄 백그라운드 작업 상태
  const [activeTasks, setActiveTasks] = useState<TaskStatus[]>([])
  const [closedTaskIds, setClosedTaskIds] = useState<Set<string>>(new Set())
  const [refreshCounter, setRefreshCounter] = useState(0)

  // 🆕 전역 알림(Toast) 상태
  const [toasts, setToasts] = useState<ToastMessage[]>([])

  const addToast = (message: string, type: 'success' | 'error' | 'info' = 'success') => {
    const id = Math.random().toString(36).substr(2, 9)
    setToasts(prev => [...prev, { id, message, type }])
  }

  const removeToast = (id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }

  // 🔄 백그라운드 작업 폴링 (Polling)
  useEffect(() => {
    if (!isConnected) return;

    const pollTasks = async () => {
      try {
        const res = await fetch(`${API_URL}/processing/list`);
        if (!res.ok) return;
        const tasks: TaskStatus[] = await res.json();

        // 필터링: 사용자가 이미 닫은 작업은 제외
        const visibleTasks = tasks.filter(t => !closedTaskIds.has(t.id));

        // 상태 변경 감지 및 알림
        tasks.forEach(task => {
          setActiveTasks(prev => {
            const prevTask = prev.find(t => t.id === task.id);
            if (task.status === 'completed' && (!prevTask || prevTask.status !== 'completed')) {
              addToast(`'${task.filename || task.doc_name || '문서'}' 처리가 완료되었습니다.`, 'success');
              setRefreshCounter(c => c + 1);
            } else if (task.status === 'error' && (!prevTask || prevTask.status !== 'error')) {
              addToast(`'${task.filename || task.doc_name || '문서'}' 처리 중 오류가 발생했습니다.`, 'error');
            }
            return visibleTasks; // 항상 최신 목록으로 교체
          });
        });

      } catch (err) {
        console.error('Task polling error:', err);
      }
    };

    const interval = setInterval(pollTasks, 3000);
    pollTasks();

    return () => clearInterval(interval);
  }, [isConnected, closedTaskIds]);

  const handleCloseTask = (id: string) => {
    setClosedTaskIds(prev => new Set(prev).add(id));
    setActiveTasks(prev => prev.filter(t => t.id !== id));
  };
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

  const handleDocumentSelect = async (docId: string, docType?: string, clause?: string) => {
    setSelectedDocument(docId)
    setSelectedClause(clause || null)
    setIsEditing(false)
    setIsOnlyOfficeMode(false)
    setOnlyOfficeConfig(null)
    setDocumentContent(null)

    // PDF → 텍스트 content 불러와서 기존 renderDocument()로 표시
    if (docType?.toLowerCase() === 'pdf') {
      try {
        const response = await fetch(`${API_URL}/rag/document/${docId}/content`)
        if (response.ok) {
          const data = await response.json()
          setDocumentContent(data.content || '')
          setEditedContent(data.content || '')
        }
      } catch (error) {
        console.error('PDF content 로드 오류:', error)
      }
      return
    }

    // DOCX → OnlyOffice
    if (docType?.toLowerCase() !== 'docx') {
      // docx가 아닌 기타 파일 → 텍스트 컨텐트로 표시
      try {
        const response = await fetch(`${API_URL}/rag/document/${docId}/content`)
        if (response.ok) {
          const data = await response.json()
          setDocumentContent(data.content || '')
          setEditedContent(data.content || '')
        }
      } catch (error) {
        console.error('문서 content 로드 오류:', error)
      }
      return
    }

    try {
      setIsOnlyOfficeMode(true)
      setOnlyOfficeEditorMode('view')

      const response = await fetch(`${API_URL}/onlyoffice/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          doc_name: docId,
          user_name: user?.username || 'Anonymous',
          mode: 'view',
        }),
      })

      if (response.ok) {
        setIsOnlyOfficeMode(true)
        setOnlyOfficeEditorMode('view')
      } else {
        console.error('OnlyOffice 설정 가져오기 실패')
        setIsOnlyOfficeMode(false)
      }
    } catch (error) {
      console.error('OnlyOffice 초기화 오류:', error)
      setIsOnlyOfficeMode(false)
    }
  }

  const handleCloseViewer = () => {
    setSelectedDocument(null)
    setSelectedClause(null)
    setDocumentContent(null)
    setIsOnlyOfficeMode(false)
    setOnlyOfficeConfig(null)
    setIsEditing(false)
  }

  const handleSaveDocument = async () => {
    if (!selectedDocument) return
    try {
      const response = await fetch(`${API_URL}/rag/document/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_name: selectedDocument, content: editedContent }),
      })
      if (response.ok) {
        setIsEditing(false)
        addToast(`문서 수정 요청이 접수되었습니다. 백그라운드에서 처리가 진행됩니다.`, 'info')
      } else {
        const errorData = await response.json()
        addToast(`저장 요청 실패: ${errorData.detail || '알 수 없는 오류'}`, 'error')
      }
    } catch {
      addToast('저장 중 오류가 발생했습니다.', 'error')
    }
  }

  const handleOpenInEditor = async (docId: string) => {
    try {
      setSelectedDocument(docId)
      setIsOnlyOfficeMode(true)
      setOnlyOfficeEditorMode('edit')

      // OnlyOffice 설정 가져오기
      const response = await fetch(`${API_URL}/onlyoffice/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          doc_name: docId,
          user_name: user?.username || 'Anonymous',
          mode: 'edit',
        }),
      })

      if (response.ok) {
        const data = await response.json()
        setOnlyOfficeConfig(data.config)
        setOnlyOfficeServerUrl(data.onlyoffice_server_url)
      } else {
        console.error('OnlyOffice 설정 가져오기 실패')
        setIsOnlyOfficeMode(false)
      }
    } catch (error) {
      console.error('OnlyOffice 초기화 오류:', error)
      setIsOnlyOfficeMode(false)
    }
  }

  const handleSwitchToEditMode = async () => {
    if (!selectedDocument) return
    try {
      setOnlyOfficeEditorMode('edit')

      // OnlyOffice 편집 모드 설정 가져오기
      const response = await fetch(`${API_URL}/onlyoffice/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          doc_name: selectedDocument,
          user_name: user?.username || 'Anonymous',
          mode: 'edit',
        }),
      })

      if (response.ok) {
        const data = await response.json()
        setOnlyOfficeConfig(data.config)
        setOnlyOfficeServerUrl(data.onlyoffice_server_url)
      } else {
        console.error('편집 모드 전환 실패')
      }
    } catch (error) {
      console.error('편집 모드 전환 오류:', error)
    }
  }

  const handleCompare = async (docName: string, v1: string, v2: string) => {
    try {
      const response = await fetch(`${API_URL}/rag/document/${docName}/compare?v1=${v1}&v2=${v2}`)
      if (response.ok) {
        const data = await response.json()
        setDiffData(data)
        setIsComparing(true)
      }
    } catch (error) {
      console.error('버전 비교 실패:', error)
    }
  }

  // 미인증 → 로그인 모달
  if (!isAuthenticated) {
    return <AuthModal onLogin={login} onRegister={register} />
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
              {isOnlyOfficeMode && onlyOfficeEditorMode === 'view' ? (
                <button
                  className="bg-dark-hover border border-dark-border text-accent py-1 px-3 text-[11px] rounded cursor-pointer transition-all duration-200 hover:bg-dark-border hover:border-txt-secondary"
                  onClick={handleSwitchToEditMode}
                >
                  수정
                </button>
              ) : !isEditing && !isOnlyOfficeMode ? (
                <button
                  className="bg-dark-hover border border-dark-border text-accent py-1 px-3 text-[11px] rounded cursor-pointer transition-all duration-200 hover:bg-dark-border hover:border-txt-secondary"
                  onClick={() => setIsEditing(true)}
                >
                  수정
                </button>
              ) : isEditing && !isOnlyOfficeMode ? (
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
              ) : null}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[12px] text-txt-secondary">
            <span className="text-accent-blue font-medium">{user?.name || user?.username}</span>
            {user?.dept && <span className="text-txt-muted ml-1">({user.dept})</span>}
          </span>
          <button
            onClick={logout}
            className="bg-transparent border border-dark-border text-txt-muted py-0.5 px-2 text-[11px] rounded cursor-pointer transition-all duration-200 hover:bg-dark-hover hover:text-[#f48771] hover:border-[#f48771]/30"
          >
            로그아웃
          </button>
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
              onOpenInEditor={handleOpenInEditor}
              refreshCounter={refreshCounter}
            />
          )}
          {activePanel === 'history' && (
            <ChangeHistoryPanel
              onCompare={handleCompare}
              selectedDocName={selectedDocument}
            />
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
          ) : selectedDocument && (documentContent || isOnlyOfficeMode) ? (
            <DocumentViewer
              selectedDocument={selectedDocument}
              targetClause={selectedClause}
              documentContent={documentContent}
              isEditing={isEditing}
              editedContent={editedContent}
              setEditedContent={setEditedContent}
              isOnlyOfficeMode={isOnlyOfficeMode}
              onlyOfficeEditorMode={onlyOfficeEditorMode}
              onlyOfficeConfig={onlyOfficeConfig}
              onlyOfficeServerUrl={onlyOfficeServerUrl}
              onClose={handleCloseViewer}
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

      {/* 백그라운드 작업 알림창 및 토스트 */}
      <Toast toasts={toasts} onRemove={removeToast} />
      <TaskNotification tasks={activeTasks} onCloseTask={handleCloseTask} />
    </div>
  )
}

export default App
