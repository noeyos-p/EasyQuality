export const SCORE_COLORS: Record<number, string> = {
  5: 'bg-[#22D142] text-black',
  4: 'bg-[#85E89D] text-black',
  3: 'bg-[#FFD700] text-black',
  2: 'bg-[#FFA500] text-black',
  1: 'bg-[#FF4444] text-white',
}

export interface RDBVerification {
  has_citations: boolean
  total_citations: number
  verified_citations: number
  incorrect_citations: string[]
  accuracy_rate: number
  verification_details: string
}

export interface EvaluationScore {
  score: number
  reasoning: string
  rdb_verification?: RDBVerification
}

export interface EvaluationScores {
  faithfulness?: EvaluationScore
  groundness?: EvaluationScore
  relevancy?: EvaluationScore
  correctness?: EvaluationScore
  average_score?: number
}

export interface ChatMessage {
  id?: string // 🆕 메시지 고유 식별자
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  thoughtProcess?: string
  thinkingTime?: number
  evaluation_scores?: EvaluationScores
  isWaiting?: boolean  // 🆕 레거시 호환용 (제거 예정)
  queuePosition?: number // 🆕 대기 순번
  status?: 'waiting' | 'processing' | 'completed' | 'error' // 🆕 상세 상태
}

export const API_URL = 'http://localhost:8000'
