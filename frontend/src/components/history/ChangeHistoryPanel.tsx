import { useState, useEffect } from 'react';
import { API_URL } from '../../types';

interface VersionRecord {
  id: number;
  doc_name: string;
  version: string;
  status: string;
  created_at: string;
  effective_at?: string;
}

interface ChangeHistoryPanelProps {
  onCompare?: (docName: string, v1: string, v2: string) => void;
  selectedDocName: string | null;
}

export default function ChangeHistoryPanel({ onCompare, selectedDocName }: ChangeHistoryPanelProps) {
  const [history, setHistory] = useState<VersionRecord[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedVersions, setSelectedVersions] = useState<string[]>([]);

  useEffect(() => {
    fetchHistory();
  }, [selectedDocName]);

  const fetchHistory = async () => {
    setIsLoading(true);
    try {
      const url = selectedDocName
        ? `${API_URL}/rag/document/${encodeURIComponent(selectedDocName)}/versions`
        : `${API_URL}/rag/changes`; // Fallback for global history

      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        setHistory(data.versions || data.changes || []);
      }
    } catch (error) {
      console.error('이력 조회 실패:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleVersionSelection = (version: string) => {
    setSelectedVersions(prev => {
      if (prev.includes(version)) {
        return prev.filter(v => v !== version);
      }
      if (prev.length >= 2) {
        return [prev[1], version]; // Keep existing logic: latest 2
      }
      return [...prev, version].sort().reverse(); // Sort descending
    });
  };

  const handleCompareClick = () => {
    if (selectedVersions.length === 2 && onCompare && selectedDocName) {
      // v1 (older), v2 (newer)
      const sorted = [...selectedVersions].sort();
      onCompare(selectedDocName, sorted[0], sorted[1]);
    }
  };

  return (
    <div className="w-full bg-dark-light border-r border-dark-border flex flex-col h-full overflow-hidden">
      <div className="px-4 py-3 border-b border-dark-border flex justify-between items-center bg-dark-deeper">
        <h2 className="text-[13px] font-semibold text-txt-primary m-0 uppercase tracking-[0.5px]">
          {selectedDocName ? `${selectedDocName} 이력` : '최근 변경 이력'}
        </h2>
        {selectedDocName && selectedVersions.length === 2 && (
          <button
            onClick={handleCompareClick}
            className="bg-accent text-black text-[11px] font-bold py-1 px-3 rounded hover:bg-accent-hover transition-all"
          >
            비교하기
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {!selectedDocName ? (
          <div className="text-center text-txt-secondary py-10 px-5 text-[12px]">
            <p>문서를 먼저 선택하면 버전 간 비교가 가능합니다.</p>
          </div>
        ) : isLoading ? (
          <div className="text-center text-txt-secondary py-10 px-5 text-[12px]">
            <p>이력을 불러오는 중...</p>
          </div>
        ) : history.length === 0 ? (
          <div className="text-center text-txt-secondary py-10 px-5 text-[12px]">
            <p>조회된 이력이 없습니다.</p>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {history.map((record) => (
              <div
                key={record.id}
                className={`group relative bg-[#1e1e1e] rounded-md p-3 border border-transparent transition-all cursor-pointer hover:border-dark-border ${selectedVersions.includes(record.version) ? 'border-accent-blue bg-accent-blue/5' : ''}`}
                onClick={() => toggleVersionSelection(record.version)}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <div className={`w-4 h-4 rounded border flex items-center justify-center transition-colors ${selectedVersions.includes(record.version) ? 'bg-accent-blue border-accent-blue' : 'border-dark-border bg-black/20'}`}>
                      {selectedVersions.includes(record.version) && <span className="text-[10px] text-white">✓</span>}
                    </div>
                    <span className="text-accent font-mono font-bold text-[13px]">v{record.version}</span>
                  </div>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded ${record.status === '사용중' ? 'bg-green-900/40 text-green-400' : 'bg-gray-800 text-gray-400'}`}>
                    {record.status}
                  </span>
                </div>
                <div className="text-txt-secondary text-[11px] mt-2">
                  <span>{new Date(record.created_at).toLocaleString('ko-KR')}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {selectedDocName && selectedVersions.length > 0 && (
        <div className="p-3 bg-dark-deeper border-t border-dark-border">
          <p className="text-[11px] text-txt-secondary text-center m-0">
            {selectedVersions.length === 1
              ? '비교할 다른 버전을 하나 더 선택하세요.'
              : `v${selectedVersions[0]}와 v${selectedVersions[1]} 선택됨`}
          </p>
        </div>
      )}
    </div>
  );
}
