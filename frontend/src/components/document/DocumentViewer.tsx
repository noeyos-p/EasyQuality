import React from 'react'
import { API_URL } from '../../types'

interface DocumentViewerProps {
  selectedDocument: string
  targetClause?: string | null
  documentContent: string | null
  isEditing: boolean
  editedContent: string
  setEditedContent: (v: string) => void
  isOnlyOfficeMode?: boolean
  onlyOfficeEditorMode?: 'view' | 'edit'
  onlyOfficeConfig?: object | null
  onlyOfficeServerUrl?: string
  onClose?: () => void
}

export default function DocumentViewer({
  selectedDocument,
  targetClause = null,
  documentContent,
  isEditing,
  editedContent,
  setEditedContent,
  isOnlyOfficeMode = false,
  onlyOfficeEditorMode = 'view',
  onlyOfficeConfig = null,
  onlyOfficeServerUrl = '',
  onClose,
}: DocumentViewerProps) {
  const [isDownloadOpen, setIsDownloadOpen] = React.useState(false)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const editorInstanceRef = React.useRef<any>(null)
  const htmlRenderRef = React.useRef<HTMLDivElement | null>(null)
  const editablePageRef = React.useRef<HTMLDivElement | null>(null)

  void onlyOfficeEditorMode

  React.useEffect(() => {
    if (!targetClause || isEditing || isOnlyOfficeMode || !documentContent) return

    const normalized = targetClause.trim()
    if (!normalized) return

    const candidates = normalized
      .split(',')
      .map(v => v.trim().replace(/\.+$/, ""))
      .filter(Boolean)
      // 더 구체적인 조항(깊이/길이)이 먼저 매칭되도록 정렬
      .sort((a, b) => {
        const depthDiff = b.split('.').length - a.split('.').length
        if (depthDiff !== 0) return depthDiff
        return b.length - a.length
      })
    const timer = window.setTimeout(() => {
      const root = htmlRenderRef.current
      if (!root) return

      let targetEl: HTMLElement | null = null
      for (const clause of candidates) {
        const escaped = (window as any).CSS?.escape ? (window as any).CSS.escape(clause) : clause.replace(/"/g, '\\"')
        const found = root.querySelector(`[data-clause="${escaped}"]`) as HTMLElement | null
        if (found) {
          targetEl = found
          break
        }
      }

      if (!targetEl) return

      targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' })
      targetEl.classList.add('ring-2', 'ring-accent-blue', 'rounded')
      window.setTimeout(() => {
        targetEl?.classList.remove('ring-2', 'ring-accent-blue', 'rounded')
      }, 2200)
    }, 80)

    return () => window.clearTimeout(timer)
  }, [targetClause, selectedDocument, documentContent, isEditing, isOnlyOfficeMode])

  // OnlyOffice 에디터 초기화
  React.useEffect(() => {
    if (!isOnlyOfficeMode || !onlyOfficeConfig || !onlyOfficeServerUrl) return

    const editorContainerId = 'onlyoffice-editor'
    const scriptSrc = `${onlyOfficeServerUrl}/web-apps/apps/api/documents/api.js`

    const initEditor = () => {
      if (editorInstanceRef.current) {
        try { editorInstanceRef.current.destroyEditor() } catch (_) { }
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

  const ensureHtml2PdfLoaded = async () => {
    if ((window as any).html2pdf) return
    await new Promise<void>((resolve, reject) => {
      const existing = document.querySelector('script[data-html2pdf="true"]') as HTMLScriptElement | null
      if (existing) {
        existing.addEventListener('load', () => resolve(), { once: true })
        existing.addEventListener('error', () => reject(new Error('html2pdf 로드 실패')), { once: true })
        return
      }
      const script = document.createElement('script')
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js'
      script.async = true
      script.dataset.html2pdf = 'true'
      script.onload = () => resolve()
      script.onerror = () => reject(new Error('html2pdf 로드 실패'))
      document.head.appendChild(script)
    })
  }

  const ensureHtmlDocxLoaded = async () => {
    if ((window as any).htmlDocx) return
    await new Promise<void>((resolve, reject) => {
      const existing = document.querySelector('script[data-htmldocx="true"]') as HTMLScriptElement | null
      if (existing) {
        existing.addEventListener('load', () => resolve(), { once: true })
        existing.addEventListener('error', () => reject(new Error('html-docx 로드 실패')), { once: true })
        return
      }
      const script = document.createElement('script')
      script.src = 'https://unpkg.com/html-docx-js/dist/html-docx.js'
      script.async = true
      script.dataset.htmldocx = 'true'
      script.onload = () => resolve()
      script.onerror = () => reject(new Error('html-docx 로드 실패'))
      document.head.appendChild(script)
    })
  }

  const cloneWithInlineStyles = (root: HTMLElement): HTMLElement => {
    const clonedRoot = root.cloneNode(true) as HTMLElement
    const walk = (src: Element, dst: Element) => {
      const computed = window.getComputedStyle(src as HTMLElement)
      let cssText = ''
      for (const prop of Array.from(computed)) {
        cssText += `${prop}:${computed.getPropertyValue(prop)};`
      }
      (dst as HTMLElement).setAttribute('style', cssText)

      const srcChildren = Array.from(src.children)
      const dstChildren = Array.from(dst.children)
      for (let i = 0; i < srcChildren.length; i += 1) {
        if (dstChildren[i]) walk(srcChildren[i], dstChildren[i])
      }
    }
    walk(root, clonedRoot)
    return clonedRoot
  }

  const handleDownload = async (format: 'pdf' | 'docx' | 'md') => {
    try {
      // PDF는 프론트 렌더 그대로 다운로드 시도
      if (format === 'pdf' && !isOnlyOfficeMode && !isEditing && htmlRenderRef.current) {
        try {
          await ensureHtml2PdfLoaded()
          const html2pdf = (window as any).html2pdf
          if (html2pdf) {
            const opt = {
              margin: [0, 0, 0, 0],
              filename: `${selectedDocument}.pdf`,
              image: { type: 'jpeg', quality: 0.98 },
              html2canvas: { scale: 2, useCORS: true, backgroundColor: '#ffffff' },
              jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
              pagebreak: { mode: ['css', 'legacy'] }
            }
            await html2pdf().set(opt).from(htmlRenderRef.current).save()
            setIsDownloadOpen(false)
            return
          }
        } catch (pdfErr) {
          console.error('html2pdf 경로 실패, 백엔드 다운로드로 폴백:', pdfErr)
        }
      }

      // DOCX도 프론트 렌더 그대로 다운로드 시도
      if (format === 'docx' && !isOnlyOfficeMode && !isEditing && htmlRenderRef.current) {
        try {
          await ensureHtmlDocxLoaded()
          const htmlDocx = (window as any).htmlDocx
          if (htmlDocx?.asBlob) {
            const renderedClone = cloneWithInlineStyles(htmlRenderRef.current)
            const html = `
              <!doctype html>
              <html>
                <head><meta charset="utf-8" /></head>
                <body style="margin:0; padding:0; background:#ffffff;">
                  ${renderedClone.innerHTML}
                </body>
              </html>
            `
            const blob = htmlDocx.asBlob(html, {
              orientation: 'portrait',
              margins: { top: 720, right: 720, bottom: 720, left: 720 }
            })
            const url = window.URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = `${selectedDocument}.docx`
            document.body.appendChild(a)
            a.click()
            document.body.removeChild(a)
            window.URL.revokeObjectURL(url)
            setIsDownloadOpen(false)
            return
          }
        } catch (docxErr) {
          console.error('html-docx 경로 실패, 백엔드 다운로드로 폴백:', docxErr)
        }
      }

      const token = localStorage.getItem('auth_token')
      const response = await fetch(`${API_URL}/rag/document/download/${selectedDocument}?format=${format}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      })
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
      } else {
        const errorText = await response.text()
        console.error(`다운로드 실패(${response.status}):`, errorText)
      }
    } catch (error) {
      console.error('다운로드 실패:', error)
    }
    setIsDownloadOpen(false)
  }

  const renderDocument = (sourceContent: string | null, editable: boolean = false) => {
    if (!sourceContent) return null

    const lines = sourceContent
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

      if (/^\*\*\*END OF DOCUMENT\*\*\*/.test(trimmedLine)) { endOfDocumentReached = true; return }
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

      if (isHeader && !sectionMatch) {
        if (!firstHeaderBlockPassed) { inHeaderBlock = true }
        else { return }
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
        const sectionStyle = { paddingLeft: `${globalDepth * indentIncrement}px` }

        if (globalDepth === 0) {
          elements.push(
            <div key={`section-${lineIdx}`} data-clause={sectionNum} className="text-[16px] font-bold mt-[60px] mb-[8px] text-black border-b border-[#e0e0e0] pb-[10px] transition-all duration-300" style={sectionStyle}>
              {displayText}
            </div>
          )
        } else {
          elements.push(
            <div key={`section-${lineIdx}`} data-clause={sectionNum} className="text-[15px] font-normal mt-[28px] mb-[8px] text-black transition-all duration-300" style={sectionStyle}>
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

      if (paragraphLines.length === 0) paragraphStartIdx = lineIdx
      paragraphLines.push(trimmedLine)
      void globalLastWasSection
    })

    flushParagraph()

    return (
      <div style={{ width: '794px' }}>
        <div className="bg-white" style={{ minHeight: '1123px', padding: '96px 90px' }}>
          <div
            ref={editable ? editablePageRef : undefined}
            className={`break-words text-black ${editable ? 'outline-none' : ''}`}
            contentEditable={editable}
            suppressContentEditableWarning={editable}
            style={editable ? { whiteSpace: 'pre-wrap' } : undefined}
            onBlur={
              editable
                ? (e) => {
                  const text = (e.currentTarget as HTMLDivElement).innerText
                  const normalized = text
                    .replace(/\u00A0/g, ' ')
                    .replace(/\n{3,}/g, '\n\n')
                    .trim()
                  setEditedContent(normalized)
                }
                : undefined
            }
          >
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
                <button className="w-full text-left px-4 py-2.5 text-[12px] text-txt-primary hover:bg-dark-hover transition-colors flex items-center gap-2" onClick={() => handleDownload('pdf')}>
                  <span className="text-red-400">📄</span> PDF Document
                </button>
                <button className="w-full text-left px-4 py-2.5 text-[12px] text-txt-primary hover:bg-dark-hover border-t border-dark-border transition-colors flex items-center gap-2" onClick={() => handleDownload('docx')}>
                  <span className="text-blue-400">📝</span> Word (.docx)
                </button>
                <button className="w-full text-left px-4 py-2.5 text-[12px] text-txt-primary hover:bg-dark-hover border-t border-dark-border transition-colors flex items-center gap-2" onClick={() => handleDownload('md')}>
                  <span className="text-green-400">markdown</span> Markdown (.md)
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 문서 내용 */}
      {isOnlyOfficeMode ? (
        <div className="flex-1 relative">
          {/* OnlyOffice 닫기 버튼 */}
          {onClose && (
            <button
              className="absolute top-3 right-3 z-[100] bg-dark-deeper/90 border border-dark-border text-txt-secondary py-1.5 px-3 rounded text-[12px] cursor-pointer hover:bg-red-900/50 hover:text-red-400 hover:border-red-400/30 transition-all duration-200 backdrop-blur-sm shadow-lg"
              onClick={onClose}
            >
              ✕ 닫기
            </button>
          )}
          <div id="onlyoffice-editor" className="w-full h-full" />
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto p-0 bg-[#c8c8c8] flex flex-col items-center gap-[30px]">
          {isEditing ? (
            <div className="w-full flex flex-col items-center gap-3 py-4">
              <div className="text-[12px] text-txt-secondary">
                화면에 보이는 형태 그대로 직접 수정할 수 있습니다.
              </div>
              {renderDocument(editedContent || documentContent, true)}
            </div>
          ) : (
            <div ref={htmlRenderRef}>
              {renderDocument(documentContent, false)}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
