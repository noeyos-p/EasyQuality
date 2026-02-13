import React from 'react'

interface DocumentViewerProps {
  selectedDocument: string
  documentContent: string | null
  isEditing: boolean
  editedContent: string
  setEditedContent: (v: string) => void
  isOnlyOfficeMode?: boolean
  onlyOfficeEditorMode?: 'view' | 'edit'
  onlyOfficeConfig?: object | null
  onlyOfficeServerUrl?: string
}

export default function DocumentViewer({
  selectedDocument,
  documentContent,
  isEditing,
  editedContent,
  setEditedContent,
  isOnlyOfficeMode = false,
  onlyOfficeEditorMode = 'view',
  onlyOfficeConfig = null,
  onlyOfficeServerUrl = '',
}: DocumentViewerProps) {
  const [isDownloadOpen, setIsDownloadOpen] = React.useState(false)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const editorInstanceRef = React.useRef<any>(null)

  // Suppress unused warning - will be used when OnlyOffice config is implemented
  void onlyOfficeEditorMode

  // OnlyOffice 에디터 초기화
  React.useEffect(() => {
    if (!isOnlyOfficeMode || !onlyOfficeConfig || !onlyOfficeServerUrl) return

    const editorContainerId = 'onlyoffice-editor'
    const scriptSrc = `${onlyOfficeServerUrl}/web-apps/apps/api/documents/api.js`

    const initEditor = () => {
      // 이전 인스턴스 파괴
      if (editorInstanceRef.current) {
        try { editorInstanceRef.current.destroyEditor() } catch (_) {}
        editorInstanceRef.current = null
      }
      const container = document.getElementById(editorContainerId)
      if (container) container.innerHTML = ''

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const DocsAPI = (window as any).DocsAPI
      if (DocsAPI) {
        editorInstanceRef.current = new DocsAPI.DocEditor(editorContainerId, onlyOfficeConfig)
      }
    }

    // 스크립트가 이미 로드됐는지 확인
    const existingScript = document.querySelector(`script[src="${scriptSrc}"]`)
    if (existingScript) {
      initEditor()
    } else {
      const script = document.createElement('script')
      script.src = scriptSrc
      script.onload = initEditor
      script.onerror = () => console.error('OnlyOffice API 스크립트 로드 실패:', scriptSrc)
      document.head.appendChild(script)
    }
  }, [isOnlyOfficeMode, onlyOfficeConfig, onlyOfficeServerUrl])

  const handleDownload = async (format: 'pdf' | 'docx' | 'md') => {
    try {
      const response = await fetch(`http://localhost:8000/rag/document/${selectedDocument}/download?format=${format}`)
      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `${selectedDocument}.${format}`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        window.URL.revokeObjectURL(url)
      }
    } catch (error) {
      console.error('다운로드 실패:', error)
    }
    setIsDownloadOpen(false)
  }

  const renderDocument = () => {
    if (!documentContent) return null

    const lines = documentContent
      .replace(/<!-- PAGE:\d+ -->/g, '')
      .split('\n')

    let globalDepth = 0
    let globalLastWasSection = false
    const indentIncrement = 12
    const elements: React.ReactElement[] = []
    let paragraphLines: string[] = []
    let paragraphStartIdx = 0
    let firstHeaderBlockPassed = false
    let inHeaderBlock = false
    let endOfDocumentReached = false

    const flushParagraph = () => {
      if (paragraphLines.length > 0) {
        const paragraphText = paragraphLines.join(' ')
        const totalPadding = globalDepth * indentIncrement

        elements.push(
          <p key={`para-${paragraphStartIdx}`} className="text-[15px] leading-[1.8] mb-[6px]" style={{ paddingLeft: `${totalPadding}px` }}>
            {paragraphText}
          </p>
        )
        paragraphLines = []
      }
    }

    lines.forEach((line, lineIdx) => {
      const trimmedLine = line.trim()

      if (/^\*\*\*END OF DOCUMENT\*\*\*/.test(trimmedLine)) {
        endOfDocumentReached = true
        return
      }
      if (endOfDocumentReached) return
      if (trimmedLine === '') return

      const sectionMatch = trimmedLine.match(/^(\d+(?:\.\d+)*)\.?\s+(.+)/)

      if (sectionMatch && /^of\s+\d+$/i.test(sectionMatch[2].trim())) return

      const isHeader = (
        /^Number:/i.test(trimmedLine) ||
        /^Version:/i.test(trimmedLine) ||
        /^Effective Date:/i.test(trimmedLine) ||
        /^Owning Department/i.test(trimmedLine) ||
        /^Title\s+GMP/i.test(trimmedLine) ||
        /^GMP 문서 체계$/i.test(trimmedLine) ||
        /for Drug Master File/i.test(trimmedLine) ||
        /품질경영실/i.test(trimmedLine)
      )

      if (isHeader) {
        if (!firstHeaderBlockPassed) {
          inHeaderBlock = true
        } else {
          return
        }
      } else if (inHeaderBlock) {
        inHeaderBlock = false
        firstHeaderBlockPassed = true
      }

      if (sectionMatch) {
        flushParagraph()

        const sectionNum = sectionMatch[1]
        const sectionText = sectionMatch[2]
        const parts = sectionNum.split('.')
        globalDepth = parts.length - 1

        const displayText = `${sectionNum} ${sectionText}`
        const sectionBasePadding = globalDepth * indentIncrement
        const sectionStyle = { paddingLeft: `${sectionBasePadding}px` }

        if (globalDepth === 0) {
          elements.push(
            <div key={`section-${lineIdx}`} className="text-[16px] font-bold mt-[60px] mb-[8px] text-black border-b border-[#e0e0e0] pb-[10px]" style={sectionStyle}>
              {displayText}
            </div>
          )
        } else {
          elements.push(
            <div key={`section-${lineIdx}`} className="text-[15px] font-normal mt-[28px] mb-[8px] text-black" style={sectionStyle}>
              {displayText}
            </div>
          )
        }
        globalLastWasSection = true
        return
      }

      if (/^={10,}/.test(trimmedLine)) {
        flushParagraph()
        elements.push(
          <div key={`separator-${lineIdx}`} className="text-[#bdc3c7] tracking-[2px] my-4 font-mono">
            {trimmedLine}
          </div>
        )
        return
      }

      const koreanChars = trimmedLine.match(/[가-힣]/g)?.length || 0
      const englishChars = trimmedLine.match(/[a-zA-Z]/g)?.length || 0
      const totalChars = koreanChars + englishChars
      const isEnglishLine = totalChars > 0 && (englishChars / totalChars) > 0.7

      if (paragraphLines.length > 0) {
        const prevLine = paragraphLines[paragraphLines.length - 1]
        const prevKorean = (prevLine.match(/[가-힣]/g)?.length || 0)
        const prevEnglish = (prevLine.match(/[a-zA-Z]/g)?.length || 0)
        const prevTotal = prevKorean + prevEnglish
        const wasPrevKorean = prevTotal > 0 && (prevKorean / prevTotal) > 0.3

        if (wasPrevKorean && isEnglishLine) {
          flushParagraph()
          paragraphStartIdx = lineIdx
        }
      }

      if (paragraphLines.length === 0) {
        paragraphStartIdx = lineIdx
      }
      paragraphLines.push(trimmedLine)

      // suppress unused warning
      void globalLastWasSection
    })

    flushParagraph()

    return (
      <div style={{ width: '794px' }}>
        <div className="bg-white" style={{ minHeight: '1123px', padding: '96px 90px' }}>
          <div className="break-words text-black">
            {elements}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* 문서 헤더 */}
      {!isOnlyOfficeMode && (
        <div className="px-6 py-4 border-b border-dark-border bg-dark-deeper flex justify-between items-center">
          <h2 className="text-[16px] font-medium text-txt-primary">{selectedDocument}</h2>
          <div className="relative">
            <button
              className="bg-accent text-black border-none py-1.5 px-4 rounded text-[12px] font-bold cursor-pointer hover:bg-accent-hover transition-all duration-200 flex items-center gap-2 shadow-lg"
              onClick={() => setIsDownloadOpen(!isDownloadOpen)}
            >
              📥 Download <span className="opacity-50">▼</span>
            </button>

            {isDownloadOpen && (
              <div className="absolute right-0 mt-2 w-40 bg-dark-light border border-dark-border rounded shadow-2xl z-50 overflow-hidden">
                <button
                  className="w-full text-left px-4 py-2.5 text-[12px] text-txt-primary hover:bg-dark-hover transition-colors flex items-center gap-2"
                  onClick={() => handleDownload('pdf')}
                >
                  <span className="text-red-400">📄</span> PDF Document
                </button>
                <button
                  className="w-full text-left px-4 py-2.5 text-[12px] text-txt-primary hover:bg-dark-hover border-t border-dark-border transition-colors flex items-center gap-2"
                  onClick={() => handleDownload('docx')}
                >
                  <span className="text-blue-400">📝</span> Word (.docx)
                </button>
                <button
                  className="w-full text-left px-4 py-2.5 text-[12px] text-txt-primary hover:bg-dark-hover border-t border-dark-border transition-colors flex items-center gap-2"
                  onClick={() => handleDownload('md')}
                >
                  <span className="text-green-400">markdown</span> Markdown (.md)
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 문서 내용 */}
      {isOnlyOfficeMode ? (
        <div
          id="onlyoffice-editor"
          className="flex-1"
          style={{ width: '100%' }}
        />
      ) : (
        <div className="flex-1 overflow-y-auto p-0 bg-[#c8c8c8] flex flex-col items-center gap-[30px]">
          {isEditing ? (
            <div className="w-full max-w-[1100px] h-[calc(100vh-120px)] bg-dark-deeper border border-dark-border rounded overflow-hidden shadow-[0_10px_30px_rgba(0,0,0,0.3)]">
              <textarea
                className="document-editor w-full h-full bg-transparent text-[#d4d4d4] border-none p-[30px] font-mono text-[14px] leading-[1.6] resize-none outline-none"
                value={editedContent}
                onChange={(e) => setEditedContent(e.target.value)}
                placeholder="문서 내용을 수정하세요..."
              />
            </div>
          ) : (
            renderDocument()
          )}
        </div>
      )}
    </div>
  )
}
