import { useState, useEffect } from 'react';
import MermaidRenderer from '../graph/MermaidRenderer';
import { API_URL } from '../../types';

interface Document {
  doc_id: string;
  title?: string;
}

interface ImpactSection {
  section_id: string;
  section_title: string;
  context: string;
}

interface Impact {
  doc_id: string;
  sections: ImpactSection[];
}

interface ImpactAnalysis {
  doc_id: string;
  impacts: Impact[];
  count: number;
  total_sections: number;
  message?: string;
}

export default function DocumentVisualizationPanel() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<string>('');
  const [mermaidCode, setMermaidCode] = useState<string>('');
  const [impactAnalysis, setImpactAnalysis] = useState<ImpactAnalysis | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // 문서 목록 로드
  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_URL}/graph/documents`);
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (error) {
      console.error('문서 목록 조회 실패:', error);
    }
  };

  // 문서 선택 시 시각화 및 영향 분석 로드
  const handleDocumentSelect = async (docId: string) => {
    if (!docId) {
      setMermaidCode('');
      setImpactAnalysis(null);
      return;
    }

    setSelectedDoc(docId);
    setIsLoading(true);

    try {
      // 시각화 데이터 로드 (Mermaid)
      const vizResponse = await fetch(`${API_URL}/graph/visualization/${docId}?format=mermaid`);
      const vizData = await vizResponse.json();
      setMermaidCode(vizData.code || '');

      // 영향 분석 로드
      const impactResponse = await fetch(`${API_URL}/graph/impact/${docId}`);
      const impactData = await impactResponse.json();
      setImpactAnalysis(impactData);
    } catch (error) {
      console.error('데이터 로드 실패:', error);
      setMermaidCode('');
      setImpactAnalysis(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full bg-dark-light border-r border-dark-border flex flex-col h-full overflow-hidden">

      {/* panel-header */}
      <div className="px-4 py-3 border-b border-dark-border">
        <h2 className="text-[13px] font-semibold text-txt-primary m-0 uppercase tracking-[0.5px]">문서 시각화</h2>
      </div>

      {/* panel-content */}
      <div className="flex-1 overflow-y-auto p-4">

        {/* document-selector */}
        <div className="mb-5">
          <label className="flex items-center gap-1.5 text-[12px] text-txt-primary mb-2 uppercase tracking-[0.5px]">
            문서 선택
          </label>
          <select
            className="w-full bg-[#3c3c3c] text-txt-primary border border-dark-border rounded py-2 px-3 text-[13px] cursor-pointer transition-colors duration-200 hover:bg-[#454545] focus:outline-none focus:border-accent-blue"
            value={selectedDoc}
            onChange={(e) => handleDocumentSelect(e.target.value)}
          >
            <option value="">문서를 선택하세요</option>
            {documents.map((doc) => (
              <option key={doc.doc_id} value={doc.doc_id}>
                {doc.doc_id} {doc.title ? `- ${doc.title}` : ''}
              </option>
            ))}
          </select>
        </div>

        {isLoading && (
          <div className="text-center text-txt-secondary py-10 px-5 text-[13px]">
            <p>데이터를 불러오는 중...</p>
          </div>
        )}

        {!isLoading && selectedDoc && (
          <>
            {/* 관계 그래프 */}
            {mermaidCode && (
              <div className="mb-6 bg-[#1e1e1e] rounded-md p-4">
                <h3 className="text-[13px] text-txt-primary mt-0 mb-4 flex items-center gap-2 uppercase tracking-[0.5px]">문서 관계 그래프</h3>
                <div className="mermaid-container bg-white rounded p-4 overflow-x-auto">
                  <MermaidRenderer chart={mermaidCode} />
                </div>
              </div>
            )}

            {/* 영향 분석 */}
            {impactAnalysis && (
              <div className="mb-6 bg-[#1e1e1e] rounded-md p-4">
                <h3 className="text-[13px] text-txt-primary mt-0 mb-4 flex items-center gap-2 uppercase tracking-[0.5px]">
                  영향 분석
                </h3>

                {impactAnalysis.count === 0 ? (
                  <p className="text-txt-secondary text-[12px] p-3 bg-[#2d2d2d] rounded border-l-[3px] border-accent-blue">
                    {impactAnalysis.message}
                  </p>
                ) : (
                  <>
                    <div className="flex gap-4 mb-4 p-3 bg-[#2d2d2d] rounded text-[12px] text-txt-primary">
                      <span>영향받는 문서: {impactAnalysis.count}개</span>
                      <span>관련 조항: {impactAnalysis.total_sections}개</span>
                    </div>

                    <div className="flex flex-col gap-3">
                      {impactAnalysis.impacts.map((impact) => (
                        <div key={impact.doc_id} className="bg-[#2d2d2d] rounded p-3 border-l-[3px] border-[#ce9178]">
                          <div className="font-semibold text-[#4ec9b0] text-[13px] mb-2">{impact.doc_id}</div>
                          <div className="flex flex-col gap-2">
                            {impact.sections.map((section, idx) => (
                              <div key={idx} className="p-2 bg-dark-light rounded-[3px]">
                                <span className="inline-block bg-accent-blue text-white py-0.5 px-2 rounded-[3px] text-[11px] font-semibold mr-2">
                                  {section.section_id}
                                </span>
                                <span className="text-txt-primary text-[12px]">{section.section_title}</span>
                                {section.context && (
                                  <p className="mt-1.5 mb-0 text-txt-secondary text-[11px] leading-[1.5]">
                                    {section.context.substring(0, 100)}...
                                  </p>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}
          </>
        )}

        {!isLoading && !selectedDoc && (
          <div className="text-center text-txt-secondary py-10 px-5 text-[13px]">
            <p>문서를 선택하여 관계 그래프와 영향 분석을 확인하세요.</p>
          </div>
        )}
      </div>
    </div>
  );
}
