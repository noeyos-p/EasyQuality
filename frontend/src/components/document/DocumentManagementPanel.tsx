import { useState, useEffect } from 'react';
import docLargeIcon from '../../assets/icons/document-manage.svg'; // Vector 21 - SOP, WI
import docSmallIcon from '../../assets/icons/document.svg';        // Vector 20 - FRM, 기타

const API_URL = 'http://localhost:8000';

interface Document {
  doc_id: string;
  doc_name?: string;
  doc_title?: string;
  doc_category?: string;
  chunk_count?: number;
  model?: string;
  collection?: string;
}

interface DocumentGroup {
  category: string;
  documents: Document[];
  expanded: boolean;
}

interface Version {
  version: string;
  created_at: string;
}

interface DocumentManagementPanelProps {
  onDocumentSelect?: (docId: string, content?: string) => void;
}

export default function DocumentManagementPanel({ onDocumentSelect }: DocumentManagementPanelProps) {
  const [groupedDocuments, setGroupedDocuments] = useState<Map<string, DocumentGroup>>(new Map());
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null);
  const [versions, setVersions] = useState<Version[]>([]);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string>('');
  const [isDeleting, setIsDeleting] = useState(false);

  // 문서 목록 로드
  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_URL}/rag/documents`);
      const data = await response.json();
      console.log('🔍 [Documents API Response]', data);
      const docs = data.documents || [];

      // 문서를 카테고리별로 그룹화
      const groups = new Map<string, DocumentGroup>();
      docs.forEach((doc: Document) => {
        const category = doc.doc_category || '기타';
        if (!groups.has(category)) {
          groups.set(category, {
            category,
            documents: [],
            expanded: true, // 기본적으로 펼쳐진 상태
          });
        }
        groups.get(category)!.documents.push(doc);
      });

      // 카테고리 순서: SOP > WI > FRM > 기타
      const sortedGroups = new Map(
        Array.from(groups.entries()).sort((a, b) => {
          const order = ['SOP', 'WI', 'FRM', '기타'];
          return order.indexOf(a[0]) - order.indexOf(b[0]);
        })
      );

      setGroupedDocuments(sortedGroups);
      return docs.length; // 문서 개수 반환
    } catch (error) {
      console.error('문서 목록 조회 실패:', error);
      return 0;
    }
  };

  // 비동기 업로드 완료 감지를 위한 폴링 로직
  const startPolling = (initialCount: number) => {
    let attempts = 0;
    const maxAttempts = 12; // 3초 * 12 = 36초

    console.log(`🚀 [Polling] 자동 갱신 시작 (현재 문서 수: ${initialCount})`);

    const intervalId = setInterval(async () => {
      attempts++;
      const currentCount = await fetchDocuments();

      console.log(`🔄 [Polling] 시도 ${attempts}/${maxAttempts} (문서 수: ${currentCount})`);

      // 새 문서가 추가되었거나 최대 시도 횟수에 도달하면 폴링 중단
      if (currentCount > initialCount || attempts >= maxAttempts) {
        clearInterval(intervalId);
        console.log(currentCount > initialCount ? "✅ [Polling] 새 문서 감지됨!" : "⏱️ [Polling] 최대 시간 도달로 종료");
      }
    }, 3000);
  };

  const toggleGroup = (category: string) => {
    setGroupedDocuments((prev) => {
      const newGroups = new Map(prev);
      const group = newGroups.get(category);
      if (group) {
        newGroups.set(category, { ...group, expanded: !group.expanded });
      }
      return newGroups;
    });
  };

  // 문서 클릭 시 최신 버전 내용 바로 표시
  const handleDocumentSelect = async (docName: string) => {
    setSelectedDoc(docName);

    try {
      const versionResponse = await fetch(`${API_URL}/rag/document/${docName}/versions`);
      const versionData = await versionResponse.json();
      const fetchedVersions: Version[] = versionData.versions || [];
      setVersions(fetchedVersions);

      // 최신 버전(첫 번째) 내용 자동 표시
      const latestVersion = fetchedVersions[0]?.version;
      await handleViewDocument(docName, latestVersion);
    } catch (error) {
      console.error('문서 로드 실패:', error);
      setVersions([]);
      await handleViewDocument(docName);
    }
  };

  // 문서 내용 보기
  const handleViewDocument = async (docName: string, version?: string) => {
    try {
      const url = version
        ? `${API_URL}/rag/document/${docName}/content?version=${version}`
        : `${API_URL}/rag/document/${docName}/content`;

      const response = await fetch(url);
      await response.json();

      // App.tsx의 뷰어에 표시
      if (onDocumentSelect) {
        onDocumentSelect(docName);
      }
    } catch (error) {
      console.error('문서 내용 조회 실패:', error);
    }
  };

  // 문서 삭제 (RDB + Weaviate + Neo4j)
  const handleDeleteDocument = async () => {
    if (!selectedDoc) return;
    if (!confirm(`"${selectedDoc}" 문서를 모든 DB에서 삭제하시겠습니까?\n(RDB, VectorDB, GraphDB 전체 삭제)`)) {
      return;
    }

    setIsDeleting(true);
    try {
      const response = await fetch(`${API_URL}/rag/document`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_name: selectedDoc, collection: 'documents', delete_from_neo4j: true }),
      });

      if (response.ok) {
        alert(`"${selectedDoc}" 삭제 완료`);
        setSelectedDoc(null);
        setVersions([]);
        fetchDocuments();
      } else {
        alert('삭제 실패');
      }
    } catch (_error) {
      alert('삭제 중 오류 발생');
    } finally {
      setIsDeleting(false);
    }
  };

  // 문서 업로드
  const handleUpload = async () => {
    if (!uploadFile) {
      alert('파일을 선택해주세요.');
      return;
    }

    setIsUploading(true);
    setUploadProgress('업로드 중...');

    const formData = new FormData();
    formData.append('file', uploadFile);
    formData.append('collection', 'documents');
    formData.append('use_langgraph', 'true');

    try {
      const response = await fetch(`${API_URL}/rag/upload`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        await response.json();
        setUploadProgress('🟢 업로드 완료! (서버 처리 중...)');

        // 현재 문서 수 확인
        const currentCount = Array.from(groupedDocuments.values()).reduce(
          (acc, group) => acc + group.documents.length,
          0
        );

        setTimeout(() => {
          setIsUploadModalOpen(false);
          setUploadFile(null);
          setUploadProgress('');
          // 🆕 비동기 처리가 완료되어 리스트에 나타날 때까지 폴링 시작
          startPolling(currentCount);
        }, 1500);
      } else {
        setUploadProgress('🔴 업로드 실패');
      }
    } catch (error) {
      console.error('업로드 실패:', error);
      setUploadProgress('🔴 업로드 중 오류 발생');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="w-full bg-dark-light border-r border-dark-border flex flex-col h-full overflow-hidden">

      {/* panel-header */}
      <div className="px-4 py-3 border-b border-dark-border flex justify-between items-center">
        <h2 className="text-[13px] font-semibold text-txt-primary m-0 uppercase tracking-[0.5px]">문서 관리</h2>

        {/* header-actions */}
        <div className="flex gap-1.5 items-center">
          {/* btn-delete-doc */}
          <button
            className="bg-dark-border text-txt-primary border-none py-1.5 px-2.5 rounded text-[12px] cursor-pointer transition-colors duration-200 disabled:opacity-40 disabled:cursor-not-allowed hover:enabled:bg-red-700 hover:enabled:text-white"
            onClick={handleDeleteDocument}
            disabled={!selectedDoc || isDeleting}
            title={selectedDoc ? `"${selectedDoc}" 삭제` : '문서를 선택하세요'}
          >
            {isDeleting ? '삭제 중...' : '- 삭제'}
          </button>

          {/* btn-upload */}
          <button
            className="bg-accent-blue text-white border-none py-1.5 px-3 rounded text-[12px] cursor-pointer flex items-center gap-1.5 transition-colors duration-200 hover:bg-[#1177bb]"
            onClick={() => setIsUploadModalOpen(true)}
          >
            + 업로드
          </button>
        </div>
      </div>

      {/* panel-content */}
      <div className="flex-1 overflow-y-auto p-2">

        {/* document-list */}
        <div className="mb-4">
          <h3 className="text-[12px] text-txt-primary mt-0 mb-2 px-2 uppercase tracking-[0.5px]">문서 목록</h3>

          {groupedDocuments.size === 0 ? (
            <p className="text-txt-secondary text-[12px] p-2 text-center">문서가 없습니다.</p>
          ) : (
            Array.from(groupedDocuments.values()).map((group) => (
              <div key={group.category} className="mb-1">

                {/* folder-header */}
                <div
                  className="flex items-center gap-1.5 py-1.5 px-2 cursor-pointer rounded transition-colors duration-200 select-none hover:bg-dark-hover"
                  onClick={() => toggleGroup(group.category)}
                >
                  <img
                    src={docLargeIcon}
                    alt="folder"
                    className="w-4 h-4 flex-shrink-0"
                    style={{ filter: 'brightness(0) invert(0.75)' }}
                  />
                  <span className="flex-1 text-[13px] font-semibold text-txt-primary">{group.category}</span>
                  <span className="text-[11px] text-txt-secondary">({group.documents.length})</span>
                </div>

                {/* folder-content */}
                {group.expanded && (
                  <div className="ml-5 border-l border-dark-border pl-1">
                    {group.documents.map((doc, idx) => (
                      <div
                        key={idx}
                        className={`flex items-center py-1.5 px-2 rounded cursor-pointer transition-colors duration-200 hover:bg-dark-hover ${selectedDoc === doc.doc_id ? 'bg-dark-active' : ''}`}
                        draggable={true}
                        onDragStart={(e) => {
                          e.dataTransfer.setData('text/plain', doc.doc_id);
                          e.dataTransfer.effectAllowed = 'copy';
                        }}
                      >
                        {/* document-info */}
                        <div
                          className="flex items-center gap-1.5 text-txt-primary text-[12px] flex-1"
                          onClick={() => handleDocumentSelect(doc.doc_id)}
                        >
                          <img
                            src={docSmallIcon}
                            alt="document"
                            className="w-3.5 h-3.5 flex-shrink-0"
                            style={{ filter: 'brightness(0) invert(0.7)' }}
                          />
                          <span>{doc.doc_id}</span>
                          {doc.chunk_count && (
                            <span className="text-txt-secondary text-[11px] ml-1">({doc.chunk_count}개)</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        {selectedDoc && versions.length > 0 && (
          <div className="mb-4">
            <h3 className="text-[12px] text-txt-primary mt-0 mb-2 px-2 uppercase tracking-[0.5px]">버전 이력</h3>
            {versions.map((ver) => (
              <div
                key={ver.version}
                className="flex justify-between items-center py-1.5 px-2 rounded transition-colors duration-200 hover:bg-dark-hover"
              >
                {/* version-info */}
                <div className="flex items-center gap-2 text-txt-primary text-[12px]">
                  <span>v{ver.version}</span>
                  <span className="text-txt-secondary text-[11px]">{new Date(ver.created_at).toLocaleDateString()}</span>
                </div>
                {/* btn-icon */}
                <button
                  className="bg-transparent border-none text-txt-primary cursor-pointer p-1 rounded-[3px] flex items-center justify-center transition-all duration-200 hover:bg-dark-border hover:text-txt-white"
                  onClick={() => handleViewDocument(selectedDoc, ver.version)}
                  title="이 버전 보기"
                >
                  보기
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* modal-overlay */}
      {isUploadModalOpen && (
        <div
          className="fixed inset-0 bg-black/70 flex items-center justify-center z-[1000]"
          onClick={() => setIsUploadModalOpen(false)}
        >
          {/* modal-content */}
          <div
            className="bg-[#2d2d2d] border border-dark-border rounded-lg p-6 min-w-[400px] shadow-[0_4px_16px_rgba(0,0,0,0.5)]"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="mt-0 mb-4 text-txt-primary text-[16px]">문서 업로드</h3>

            <input
              type="file"
              accept=".pdf"
              className="w-full mb-4 text-txt-primary"
              onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
              disabled={isUploading}
            />

            {/* upload-progress */}
            {uploadProgress && (
              <p className="text-[#4ec9b0] text-[12px] mb-4">{uploadProgress}</p>
            )}

            {/* modal-actions */}
            <div className="flex gap-2 justify-end">
              <button
                className="py-2 px-4 border-none rounded cursor-pointer text-[13px] transition-colors duration-200 bg-accent-blue text-white disabled:opacity-50 disabled:cursor-not-allowed hover:enabled:bg-[#1177bb]"
                onClick={handleUpload}
                disabled={isUploading || !uploadFile}
              >
                업로드
              </button>
              <button
                className="py-2 px-4 border-none rounded cursor-pointer text-[13px] transition-colors duration-200 bg-dark-border text-txt-primary disabled:opacity-50 disabled:cursor-not-allowed hover:enabled:bg-[#4e4e4e]"
                onClick={() => setIsUploadModalOpen(false)}
                disabled={isUploading}
              >
                취소
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
