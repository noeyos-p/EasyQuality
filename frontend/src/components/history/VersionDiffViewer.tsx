interface DiffItem {
    clause: string
    change_type: 'ADDED' | 'DELETED' | 'MODIFIED' | 'UNCHANGED'
    v1_content: string | null
    v2_content: string | null
}

interface VersionDiffViewerProps {
    diffData: {
        doc_name: string
        v1: string
        v2: string
        diffs: DiffItem[]
    }
    onClose: () => void
}

export default function VersionDiffViewer({ diffData, onClose }: VersionDiffViewerProps) {
    const getChangeStyle = (type: string) => {
        switch (type) {
            case 'ADDED': return 'bg-green-900/20 border-l-4 border-green-500'
            case 'DELETED': return 'bg-red-900/20 border-l-4 border-red-500 text-gray-400'
            case 'MODIFIED': return 'bg-yellow-900/20 border-l-4 border-yellow-500'
            default: return ''
        }
    }

    const getLabel = (type: string) => {
        switch (type) {
            case 'ADDED': return <span className="text-[10px] bg-green-500 text-black px-1 rounded font-bold">ADDED</span>
            case 'DELETED': return <span className="text-[10px] bg-red-500 text-white px-1 rounded font-bold">DELETED</span>
            case 'MODIFIED': return <span className="text-[10px] bg-yellow-500 text-black px-1 rounded font-bold">MODIFIED</span>
            default: return null
        }
    }

    return (
        <div className="flex flex-col h-full bg-[#1e1e1e] overflow-hidden">
            <div className="flex justify-between items-center px-6 py-4 bg-dark-deeper border-b border-dark-border">
                <div className="flex flex-col">
                    <h2 className="text-[16px] font-medium text-txt-primary m-0">버전 비교: {diffData.doc_name}</h2>
                    <p className="text-[12px] text-txt-secondary mt-1">
                        <span className="text-red-400 font-mono">v{diffData.v1}</span> 와 <span className="text-green-400 font-mono">v{diffData.v2}</span> 비교
                    </p>
                </div>
                <button
                    onClick={onClose}
                    className="bg-dark-light text-txt-primary border border-dark-border py-1.5 px-4 rounded text-[12px] hover:bg-dark-hover transition-colors"
                >
                    상세 보기로 돌아가기
                </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6">
                <div className="max-w-[1200px] mx-auto space-y-4">
                    {diffData.diffs.map((item, idx) => (
                        <div key={idx} className={`p-4 rounded-md transition-all ${getChangeStyle(item.change_type)}`}>
                            <div className="flex items-center gap-3 mb-2">
                                <span className="text-accent font-mono font-bold text-[14px]">{item.clause}</span>
                                {getLabel(item.change_type)}
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {(item.change_type === 'DELETED' || item.change_type === 'MODIFIED') && (
                                    <div className="flex flex-col gap-1">
                                        <span className="text-[11px] text-red-400 font-medium">v{diffData.v1} (삭제됨/이전)</span>
                                        <div className="bg-black/30 p-3 rounded text-[13px] leading-[1.6] whitespace-pre-wrap line-through opacity-60">
                                            {item.v1_content}
                                        </div>
                                    </div>
                                )}

                                {(item.change_type === 'ADDED' || item.change_type === 'MODIFIED') && (
                                    <div className="flex flex-col gap-1">
                                        <span className="text-[11px] text-green-400 font-medium">v{diffData.v2} (추가됨/현재)</span>
                                        <div className="bg-black/30 p-3 rounded text-[13px] leading-[1.6] whitespace-pre-wrap font-medium">
                                            {item.v2_content}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}

                    {diffData.diffs.length === 0 && (
                        <div className="flex flex-col items-center justify-center py-20 text-txt-secondary">
                            <span className="text-[48px] opacity-30 mb-4">🔍</span>
                            <p>두 버전 사이에 변경된 내용이 없습니다.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
