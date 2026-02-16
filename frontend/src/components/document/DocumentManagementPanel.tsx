import { useState, useEffect } from 'react';
import docLargeIcon from '../../assets/icons/document-manage.svg'; // Vector 21 - SOP, WI
import docSmallIcon from '../../assets/icons/document.svg';        // Vector 20 - FRM, 기타
import { API_URL } from '../../types';

interface Document {
  doc_id: string;
  doc_name?: string;
  doc_title?: string;
  doc_category?: string;
  doc_type?: string;
  doc_format?: string;
  chunk_count?: number;
  model?: string;
  collection?: string;
  version?: string;
}

interface DocumentGroup {
  category: string;
  documents: Document[];
  expanded: boolean;
}

interface FormatGroup {
  format: string;
  label: string;
  expanded: boolean;
  categories: Map<string, DocumentGroup>;
}

interface Version {
  version: string;
  created_at: string;
}

interface DocumentManagementPanelProps {
  onDocumentSelect?: (docId: string, docType?: string) => void;
  onNotify?: (message: string, type?: 'success' | 'error' | 'info') => void;
  onOpenInEditor?: (docId: string, version?: string) => void;
}

export default function DocumentManagementPanel({ onDocumentSelect, onNotify, onOpenInEditor }: DocumentManagementPanelProps) {
  const [groupedDocuments, setGroupedDocuments] = useState<Map<string, FormatGroup>>(new Map());
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null);
  const [selectedDocs, setSelectedDocs] = useState<Set<string>>(new Set());
  const [versions, setVersions] = useState<Version[]>([]);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string>('');
  const [isDeleting, setIsDeleting] = useState(false);
  const [docxDocName, setDocxDocName] = useState<string>('');
  const [docxVersion, setDocxVersion] = useState<string>('1.0');

  const hasDocxFile = uploadFiles.some((f) => f.name.toLowerCase().endsWith('.docx'));

  const normalizeDocType = (docType?: string): string => {
    if (!docType) return 'other';
    const t = docType.toLowerCase();
    if (t.includes('pdf')) return 'pdf';
    if (t.includes('docx')) return 'docx';
    return 'other';
  };

  const getFormatFolderLabel = (format: string): string => {
    if (format === 'pdf') return 'PDF';
    if (format === 'docx') return 'DOCX';
    return '기타';
  };

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

      // 문서를 포맷(PDF/DOCX/기타) > 카테고리(SOP/WI/FRM/기타) 2단계로 그룹화
      const groups = new Map<string, FormatGroup>();
      docs.forEach((doc: Document) => {
        const format = (doc.doc_format || normalizeDocType(doc.doc_type)).toLowerCase();
        const formatLabel = getFormatFolderLabel(format);
        if (!groups.has(formatLabel)) {
          groups.set(formatLabel, {
            format,
            label: formatLabel,
            expanded: true,
            categories: new Map<string, DocumentGroup>(),
          });
        }

        const formatGroup = groups.get(formatLabel)!;
        const category = doc.doc_category || '기타';
        if (!formatGroup.categories.has(category)) {
          formatGroup.categories.set(category, {
            category,
            documents: [],
            expanded: true,
          });
        }
        formatGroup.categories.get(category)!.documents.push(doc);
      });

      // 정렬: 포맷(PDF > DOCX > 기타), 카테고리(SOP > WI > FRM > 기타)
      const sortedGroups = new Map<string, FormatGroup>();
      Array.from(groups.entries())
        .sort((a, b) => {
          const order = ['PDF', 'DOCX', '기타'];
          return order.indexOf(a[0]) - order.indexOf(b[0]);
        })
        .forEach(([key, formatGroup]) => {
          const sortedCategories = new Map(
            Array.from(formatGroup.categories.entries()).sort((a, b) => {
              const order = ['SOP', 'WI', 'FRM', '기타'];
              const ia = order.indexOf(a[0]);
              const ib = order.indexOf(b[0]);
              if (ia === -1 && ib === -1) return a[0].localeCompare(b[0]);
              if (ia === -1) return 1;
              if (ib === -1) return -1;
              return ia - ib;
            })
          );
          sortedGroups.set(key, { ...formatGroup, categories: sortedCategories });
        });

      setGroupedDocuments(sortedGroups);
    } catch (error) {
      console.error('문서 목록 조회 실패:', error);
    }
  };

  const toggleFormatGroup = (formatLabel: string) => {
    setGroupedDocuments((prev) => {
      const newGroups = new Map(prev);
      const group = newGroups.get(formatLabel);
      if (group) {
        newGroups.set(formatLabel, { ...group, expanded: !group.expanded });
      }
      return newGroups;
    });
  };

  const toggleCategoryGroup = (formatLabel: string, category: string) => {
    setGroupedDocuments((prev) => {
      const newGroups = new Map(prev);
      const formatGroup = newGroups.get(formatLabel);
      if (!formatGroup) return newGroups;

      const newCategories = new Map(formatGroup.categories);
      const catGroup = newCategories.get(category);
      if (catGroup) {
        newCategories.set(category, { ...catGroup, expanded: !catGroup.expanded });
      }
      newGroups.set(formatLabel, { ...formatGroup, categories: newCategories });
      return newGroups;
    });
  };

  // docId로 doc_type 조회
  const getDocType = (docId: string): string | undefined => {
    for (const formatGroup of groupedDocuments.values()) {
      for (const categoryGroup of formatGroup.categories.values()) {
        const doc = categoryGroup.documents.find((d) => d.doc_id === docId);
        if (doc) return normalizeDocType(doc.doc_type);
      }
    }
    return undefined;
  };

  // 문서 클릭 시 최신 버전 내용 바로 표시
  const handleDocumentSelect = async (docName: string, explicitDocType?: string) => {
    setSelectedDoc(docName);
    const docType = explicitDocType || getDocType(docName);

    try {
      const versionResponse = await fetch(`${API_URL}/rag/document/${docName}/versions`);
      const versionData = await versionResponse.json();
      const fetchedVersions: Version[] = versionData.versions || [];
      setVersions(fetchedVersions);

      const latestVersion = fetchedVersions[0]?.version;
      await handleViewDocument(docName, latestVersion, docType);
    } catch (error) {
      console.error('문서 로드 실패:', error);
      setVersions([]);
      await handleViewDocument(docName, undefined, docType);
    }
  };

  // 문서 내용 보기
  const handleViewDocument = async (docName: string, version?: string, docType?: string) => {
    // 항상 onDocumentSelect 호출 (content 조회 실패해도 뷰어는 열림)
    if (onDocumentSelect) {
      onDocumentSelect(docName, docType);
    }
    try {
      const url = version
        ? `${API_URL}/rag/document/${docName}/content?version=${version}`
        : `${API_URL}/rag/document/${docName}/content`;
      await fetch(url);
    } catch (error) {
      console.error('문서 내용 조회 실패:', error);
    }
  };

  // 문서 삭제 (RDB + Weaviate + Neo4j)
  const handleDeleteDocument = async () => {
    const targets = selectedDocs.size > 0
      ? Array.from(selectedDocs)
      : (selectedDoc ? [selectedDoc] : []);
    if (targets.length === 0) return;

    const targetText = targets.length === 1
      ? `"${targets[0]}"`
      : `${targets.length}개 문서`;

    if (!confirm(`${targetText}를 모든 DB에서 완전 삭제하시겠습니까?\n(RDB, VectorDB, GraphDB 전체 삭제 검증 포함)`)) {
      return;
    }

    setIsDeleting(true);
    try {
      const response = await fetch(`${API_URL}/rag/documents/delete-batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          doc_names: targets,
          collection: 'documents',
          delete_from_neo4j: true
        }),
      });

      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          alert(`선택한 문서 ${targets.length}개 삭제 완료`);
          if (onNotify) onNotify(`문서 ${targets.length}개 삭제 완료`, 'success');
        } else {
          const failed = data.failed_count ?? 0;
          const failedItems = Array.isArray(data.results)
            ? data.results.filter((r: any) => !r.success)
            : [];
          const failedSummary = failedItems
            .map((r: any) => {
              const details = r.details || {};
              const failedDb = ['rdb', 'weaviate', 'neo4j']
                .concat('s3_docx')
                .filter((k) => details[k] && details[k].success === false)
                .join(',');
              return `- ${r.doc_name}${failedDb ? ` (${failedDb})` : ''}`;
            })
            .join('\n');

          alert(
            `일부 삭제 실패 (${failed}개 실패)\n` +
            `${failedSummary || '실패 문서 정보를 확인할 수 없습니다.'}\n\n` +
            `백엔드 로그에서 [DELETE] 라인도 확인해주세요.`
          );
          if (onNotify) onNotify(`일부 삭제 실패 (${failed}개)`, 'error');
        }

        const removedSet = new Set(targets);
        if (selectedDoc && removedSet.has(selectedDoc)) {
          setSelectedDoc(null);
          setVersions([]);
        }
        setSelectedDocs(new Set());
        fetchDocuments();
      } else {
        const err = await response.json();
        alert(`삭제 실패: ${err.detail || '알 수 없는 오류'}`);
      }
    } catch (_error) {
      alert('삭제 중 오류 발생');
    } finally {
      setIsDeleting(false);
    }
  };

  const toggleDocChecked = (docId: string) => {
    setSelectedDocs((prev) => {
      const next = new Set(prev);
      if (next.has(docId)) next.delete(docId);
      else next.add(docId);
      return next;
    });
  };

  // 문서 업로드
  const handleUpload = async () => {
    if (uploadFiles.length === 0) {
      alert('파일을 선택해주세요.');
      return;
    }
    if (uploadFiles.length === 1 && hasDocxFile && !docxDocName.trim()) {
      alert('문서 ID를 입력해주세요. (예: EQ-SOP-00001)');
      return;
    }

    setIsUploading(true);
    setUploadProgress(`업로드 중... (0/${uploadFiles.length})`);

    try {
      const failed: string[] = [];

      for (let i = 0; i < uploadFiles.length; i += 1) {
        const file = uploadFiles[i];
        const formData = new FormData();
        formData.append('file', file);

        const isDocx = file.name.toLowerCase().endsWith('.docx');
        setUploadProgress(`업로드 중... (${i + 1}/${uploadFiles.length}) ${file.name}`);

        let response: Response;
        if (isDocx) {
          const inferredDocName = file.name.replace(/\.[^.]+$/, '').trim();
          const targetDocName = (uploadFiles.length === 1 ? docxDocName.trim() : inferredDocName) || inferredDocName;
          formData.append('doc_name', targetDocName);
          formData.append('version', docxVersion.trim() || '1.0');
          formData.append('collection', 'documents');
          response = await fetch(`${API_URL}/rag/upload-docx`, { method: 'POST', body: formData });
        } else {
          formData.append('collection', 'documents');
          formData.append('use_langgraph', 'true');
          response = await fetch(`${API_URL}/rag/upload`, { method: 'POST', body: formData });
        }

        if (!response.ok) {
          failed.push(file.name);
        }
      }

      if (failed.length === 0) {
        setUploadProgress(`업로드 완료! (${uploadFiles.length}개)`);
        if (onNotify) onNotify(`문서 ${uploadFiles.length}개 업로드 완료`, 'success');
      } else {
        setUploadProgress(`일부 업로드 실패 (${failed.length}개): ${failed.join(', ')}`);
        if (onNotify) onNotify(`일부 업로드 실패 (${failed.length}개)`, 'error');
      }

      setTimeout(() => {
        setIsUploadModalOpen(false);
        setUploadFiles([]);
        setDocxDocName('');
        setDocxVersion('1.0');
        setUploadProgress('');
        fetchDocuments();
      }, 1500);
    } catch (error) {
      console.error('업로드 실패:', error);
      setUploadProgress('업로드 중 오류 발생');
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
            disabled={(selectedDocs.size === 0 && !selectedDoc) || isDeleting}
            title={selectedDocs.size > 0 ? `${selectedDocs.size}개 문서 삭제` : (selectedDoc ? `"${selectedDoc}" 삭제` : '문서를 선택하세요')}
          >
            {isDeleting ? '삭제 중...' : `- 삭제${selectedDocs.size > 0 ? ` (${selectedDocs.size})` : ''}`}
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
            Array.from(groupedDocuments.values()).map((formatGroup) => {
              const formatCount = Array.from(formatGroup.categories.values())
                .reduce((sum, cg) => sum + cg.documents.length, 0);
              return (
              <div key={formatGroup.label} className="mb-1">

                {/* folder-header */}
                <div
                  className="flex items-center gap-1.5 py-1.5 px-2 cursor-pointer rounded transition-colors duration-200 select-none hover:bg-dark-hover"
                  onClick={() => toggleFormatGroup(formatGroup.label)}
                >
                  <img
                    src={docLargeIcon}
                    alt="folder"
                    className="w-4 h-4 flex-shrink-0"
                    style={{ filter: 'brightness(0) invert(0.75)' }}
                  />
                  <span className="flex-1 text-[13px] font-semibold text-txt-primary">{formatGroup.label}</span>
                  <span className="text-[11px] text-txt-secondary">({formatCount})</span>
                </div>

                {/* folder-content */}
                {formatGroup.expanded && (
                  <div className="ml-5 border-l border-dark-border pl-1">
                    {Array.from(formatGroup.categories.values()).map((categoryGroup) => (
                      <div key={`${formatGroup.label}-${categoryGroup.category}`} className="mb-1">
                        <div
                          className="flex items-center gap-1.5 py-1.5 px-2 cursor-pointer rounded transition-colors duration-200 select-none hover:bg-dark-hover"
                          onClick={() => toggleCategoryGroup(formatGroup.label, categoryGroup.category)}
                        >
                          <img
                            src={docLargeIcon}
                            alt="subfolder"
                            className="w-3.5 h-3.5 flex-shrink-0"
                            style={{ filter: 'brightness(0) invert(0.68)' }}
                          />
                          <span className="flex-1 text-[12px] font-semibold text-txt-primary">{categoryGroup.category}</span>
                          <span className="text-[11px] text-txt-secondary">({categoryGroup.documents.length})</span>
                        </div>

                        {categoryGroup.expanded && (
                          <div className="ml-4 border-l border-dark-border pl-1">
                            {categoryGroup.documents.map((doc, idx) => (
                              <div
                                key={`${doc.doc_id}-${idx}`}
                                className={`flex items-center py-1.5 px-2 rounded cursor-pointer transition-colors duration-200 hover:bg-dark-hover ${selectedDoc === doc.doc_id ? 'bg-dark-active' : ''}`}
                                draggable={true}
                                onDragStart={(e) => {
                                  e.dataTransfer.setData('text/plain', doc.doc_id);
                                  e.dataTransfer.effectAllowed = 'copy';
                                }}
                              >
                                <input
                                  type="checkbox"
                                  className="mr-2 accent-[#4ec9b0] cursor-pointer"
                                  checked={selectedDocs.has(doc.doc_id)}
                                  onChange={() => toggleDocChecked(doc.doc_id)}
                                  onClick={(e) => e.stopPropagation()}
                                  title="삭제 대상 선택"
                                />
                                {/* document-info */}
                                <div
                                  className="flex items-center gap-1.5 text-txt-primary text-[12px] flex-1"
                                  onClick={() => handleDocumentSelect(doc.doc_id, normalizeDocType(doc.doc_type))}
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
                                {normalizeDocType(doc.doc_type) === 'docx' && onOpenInEditor && (
                                  <button
                                    className="ml-1 bg-transparent border border-dark-border text-[#4ec9b0] text-[10px] py-0.5 px-1.5 rounded cursor-pointer transition-all duration-200 hover:bg-dark-border hover:text-white flex-shrink-0"
                                    onClick={(e) => { e.stopPropagation(); onOpenInEditor(doc.doc_id, doc.version) }}
                                    title="OnlyOffice 에디터에서 열기"
                                  >
                                    편집
                                  </button>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )})
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
                  onClick={() => handleViewDocument(selectedDoc, ver.version, getDocType(selectedDoc))}
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
              accept=".pdf,.docx"
              multiple
              className="w-full mb-4 text-txt-primary"
              onChange={(e) => setUploadFiles(Array.from(e.target.files || []))}
              disabled={isUploading}
            />

            {/* DOCX 파일일 때 추가 입력 필드 */}
            {uploadFiles.length === 1 && hasDocxFile && (
              <>
                <input
                  type="text"
                  placeholder="문서 ID (예: EQ-SOP-00001)"
                  className="w-full mb-2 px-3 py-2 bg-dark-bg border border-dark-border rounded text-txt-primary text-[13px]"
                  value={docxDocName}
                  onChange={(e) => setDocxDocName(e.target.value)}
                  disabled={isUploading}
                />
                <input
                  type="text"
                  placeholder="버전 (예: 1.0)"
                  className="w-full mb-4 px-3 py-2 bg-dark-bg border border-dark-border rounded text-txt-primary text-[13px]"
                  value={docxVersion}
                  onChange={(e) => setDocxVersion(e.target.value)}
                  disabled={isUploading}
                />
              </>
            )}

            {/* upload-progress */}
            {uploadProgress && (
              <p className="text-[#4ec9b0] text-[12px] mb-4">{uploadProgress}</p>
            )}

            {uploadFiles.length > 0 && (
              <p className="text-txt-secondary text-[11px] mb-3">
                선택된 파일: {uploadFiles.length}개
              </p>
            )}

            {/* modal-actions */}
            <div className="flex gap-2 justify-end">
              <button
                className="py-2 px-4 border-none rounded cursor-pointer text-[13px] transition-colors duration-200 bg-accent-blue text-white disabled:opacity-50 disabled:cursor-not-allowed hover:enabled:bg-[#1177bb]"
                onClick={handleUpload}
                disabled={isUploading || uploadFiles.length === 0}
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
