import { useState, useEffect } from 'react'
import { Eye, EyeOff } from 'lucide-react'
import { API_URL } from '../../types'

interface AuthModalProps {
    onLogin: (username: string, password: string) => Promise<any>
    onRegister: (username: string, password: string, name: string, email: string, dept?: string, rank?: string) => Promise<any>
}

type Mode = 'login' | 'register' | 'findId' | 'findPw'

export default function AuthModal({ onLogin, onRegister }: AuthModalProps) {
    const [mode, setMode] = useState<Mode>('login')
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState('')
    const [success, setSuccess] = useState('')

    // 드롭다운 옵션
    const [departments, setDepartments] = useState<string[]>([])
    const [ranks, setRanks] = useState<string[]>([])

    // 로그인 폼
    const [loginUsername, setLoginUsername] = useState('')
    const [loginPassword, setLoginPassword] = useState('')

    // 회원가입 폼
    const [regUsername, setRegUsername] = useState('')
    const [regPassword, setRegPassword] = useState('')
    const [regPasswordConfirm, setRegPasswordConfirm] = useState('')
    const [regName, setRegName] = useState('')
    const [regEmail, setRegEmail] = useState('')
    const [regDept, setRegDept] = useState('')
    const [regRank, setRegRank] = useState('')

    // 비밀번호 가시성
    const [showLoginPw, setShowLoginPw] = useState(false)
    const [showRegPw, setShowRegPw] = useState(false)
    const [showRegPwConfirm, setShowRegPwConfirm] = useState(false)
    const [showNewPw, setShowNewPw] = useState(false)
    const [showNewPwConfirm, setShowNewPwConfirm] = useState(false)

    // 아이디 찾기
    const [findIdName, setFindIdName] = useState('')
    const [findIdDept, setFindIdDept] = useState('')
    const [foundUsername, setFoundUsername] = useState('')

    // 비밀번호 찾기
    const [findPwUsername, setFindPwUsername] = useState('')
    const [findPwName, setFindPwName] = useState('')
    const [newPassword, setNewPassword] = useState('')
    const [newPasswordConfirm, setNewPasswordConfirm] = useState('')
    const [pwResetStep, setPwResetStep] = useState<'verify' | 'reset'>('verify')
    const [resetUserId, setResetUserId] = useState<number | null>(null)

    // 부서/직책 옵션 로드
    useEffect(() => {
        fetch(`${API_URL}/auth/options`)
            .then(res => res.json())
            .then(data => {
                setDepartments(data.departments || [])
                setRanks(data.ranks || [])
            })
            .catch(() => { })
    }, [])

    const switchMode = (newMode: Mode) => {
        setMode(newMode)
        setError('')
        setSuccess('')
        setFoundUsername('')
        setPwResetStep('verify')
        setResetUserId(null)
    }

    const inputCls = "w-full px-3 py-2 rounded text-[13px] text-txt-primary placeholder-txt-muted outline-none transition-colors duration-150 focus:border-txt-secondary"
    const inputStyle = { background: '#1F1F1F', border: '1px solid #3e3e42' }
    const selectCls = "w-full px-3 py-2 rounded text-[13px] text-txt-primary outline-none transition-colors duration-150 focus:border-txt-secondary appearance-none cursor-pointer"
    const btnCls = "w-full py-2.5 rounded text-[13px] font-medium text-dark-deeper bg-txt-primary border-none cursor-pointer transition-colors duration-150 hover:bg-[#b0b0b0] disabled:opacity-50 disabled:cursor-not-allowed"

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault()
        setError('')
        if (!loginUsername.trim()) return setError('아이디를 입력하세요.')
        if (!loginPassword) return setError('비밀번호를 입력하세요.')
        setIsLoading(true)
        try { await onLogin(loginUsername.trim(), loginPassword) }
        catch (err: any) { setError(err.message || '로그인에 실패했습니다.') }
        finally { setIsLoading(false) }
    }

    const handleRegister = async (e: React.FormEvent) => {
        e.preventDefault()
        setError(''); setSuccess('')
        if (!regUsername.trim()) return setError('아이디를 입력하세요.')
        if (regUsername.trim().length < 4) return setError('아이디는 최소 4자 이상이어야 합니다.')
        if (!regPassword) return setError('비밀번호를 입력하세요.')
        if (regPassword.length < 6) return setError('비밀번호는 최소 6자 이상이어야 합니다.')
        if (regPassword !== regPasswordConfirm) return setError('비밀번호가 일치하지 않습니다.')
        if (!regName.trim()) return setError('이름을 입력하세요.')
        if (!regEmail.trim()) return setError('이메일을 입력하세요.')
        const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/
        if (!emailRegex.test(regEmail.trim())) return setError('올바른 이메일 형식이 아닙니다.')

        setIsLoading(true)
        try {
            await onRegister(regUsername.trim(), regPassword, regName.trim(), regEmail.trim(), regDept || undefined, regRank || undefined)
            setSuccess('회원가입이 완료되었습니다! 로그인해주세요.')
            setTimeout(() => { switchMode('login'); setLoginUsername(regUsername) }, 1500)
        } catch (err: any) { setError(err.message || '회원가입에 실패했습니다.') }
        finally { setIsLoading(false) }
    }

    const handleFindId = async (e: React.FormEvent) => {
        e.preventDefault()
        setError(''); setFoundUsername('')
        if (!findIdName.trim()) return setError('이름을 입력하세요.')
        setIsLoading(true)
        try {
            const res = await fetch(`${API_URL}/auth/find-username`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: findIdName.trim(), dept: findIdDept || undefined }),
            })
            const data = await res.json()
            if (!res.ok) throw new Error(data.detail || '아이디를 찾을 수 없습니다.')
            setFoundUsername(data.username)
            setSuccess('아이디를 찾았습니다.')
        } catch (err: any) { setError(err.message) }
        finally { setIsLoading(false) }
    }

    const handleFindPwVerify = async (e: React.FormEvent) => {
        e.preventDefault()
        setError('')
        if (!findPwUsername.trim()) return setError('아이디를 입력하세요.')
        if (!findPwName.trim()) return setError('이름을 입력하세요.')
        setIsLoading(true)
        try {
            const res = await fetch(`${API_URL}/auth/verify-user`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username: findPwUsername.trim(), name: findPwName.trim() }),
            })
            const data = await res.json()
            if (!res.ok) throw new Error(data.detail || '사용자를 확인할 수 없습니다.')
            setResetUserId(data.user_id)
            setPwResetStep('reset')
            setSuccess('본인 확인 완료. 새 비밀번호를 입력하세요.')
        } catch (err: any) { setError(err.message) }
        finally { setIsLoading(false) }
    }

    const handleResetPassword = async (e: React.FormEvent) => {
        e.preventDefault()
        setError('')
        if (!newPassword) return setError('새 비밀번호를 입력하세요.')
        if (newPassword.length < 6) return setError('비밀번호는 최소 6자 이상이어야 합니다.')
        if (newPassword !== newPasswordConfirm) return setError('비밀번호가 일치하지 않습니다.')
        setIsLoading(true)
        try {
            const res = await fetch(`${API_URL}/auth/reset-password`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: resetUserId, new_password: newPassword }),
            })
            const data = await res.json()
            if (!res.ok) throw new Error(data.detail || '비밀번호 변경 실패')
            setSuccess('비밀번호가 변경되었습니다! 로그인해주세요.')
            setTimeout(() => { switchMode('login'); setLoginUsername(findPwUsername) }, 1500)
        } catch (err: any) { setError(err.message) }
        finally { setIsLoading(false) }
    }

    return (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-[#0d0d0d]">
            <div className="w-[400px] max-w-[90vw] bg-dark-deeper border border-dark-border rounded-lg overflow-hidden">
                <div className="h-[2px] bg-txt-primary" />

                {/* 헤더 */}
                <div className="px-8 pt-7 pb-1 text-center">
                    <h1 className="text-[18px] font-semibold text-txt-primary tracking-tight m-0">EasyQuality</h1>
                    <p className="text-[11px] text-txt-muted mt-1">GMP Document Management System</p>
                </div>

                {/* 탭 */}
                {(mode === 'login' || mode === 'register') && (
                    <div className="flex mx-8 mt-5 border border-dark-border rounded overflow-hidden">
                        <button className={`flex-1 py-2 text-[12px] font-medium border-none cursor-pointer transition-colors duration-150 ${mode === 'login' ? 'bg-dark-active text-txt-primary' : 'bg-dark-deeper text-txt-muted hover:text-txt-secondary'}`}
                            onClick={() => switchMode('login')}>로그인</button>
                        <button className={`flex-1 py-2 text-[12px] font-medium border-none cursor-pointer transition-colors duration-150 ${mode === 'register' ? 'bg-dark-active text-txt-primary' : 'bg-dark-deeper text-txt-muted hover:text-txt-secondary'}`}
                            onClick={() => switchMode('register')}>회원가입</button>
                    </div>
                )}

                {/* 서브 페이지 헤더 */}
                {(mode === 'findId' || mode === 'findPw') && (
                    <div className="mx-8 mt-5 flex items-center gap-2">
                        <button onClick={() => switchMode('login')} className="bg-transparent border-none text-txt-muted text-[13px] cursor-pointer hover:text-txt-primary p-0">←</button>
                        <span className="text-[13px] text-txt-primary font-medium">{mode === 'findId' ? '아이디 찾기' : '비밀번호 찾기'}</span>
                    </div>
                )}

                {/* 메시지 */}
                {error && (
                    <div className="mx-8 mt-4 px-3 py-2 rounded text-[12px] flex items-center gap-2 border-l-2 border-[#f48771]"
                        style={{ background: 'rgba(244,135,113,0.08)', color: '#f48771' }}>
                        <span>⚠</span> {error}
                    </div>
                )}
                {success && (
                    <div className="mx-8 mt-4 px-3 py-2 rounded text-[12px] flex items-center gap-2 border-l-2 border-accent"
                        style={{ background: 'rgba(34,209,66,0.08)', color: '#22D142' }}>
                        <span>✓</span> {success}
                    </div>
                )}

                {/* ───── 로그인 ───── */}
                {mode === 'login' && (
                    <form onSubmit={handleLogin} className="px-8 pt-5 pb-4">
                        <div className="mb-4">
                            <label className="block text-[11px] text-txt-secondary mb-1.5">아이디</label>
                            <input type="text" value={loginUsername} onChange={e => setLoginUsername(e.target.value)}
                                placeholder="username" autoFocus disabled={isLoading} className={inputCls} style={inputStyle} />
                        </div>
                        <div className="mb-5">
                            <label className="block text-[11px] text-txt-secondary mb-1.5">비밀번호</label>
                            <div className="relative">
                                <input type={showLoginPw ? "text" : "password"} value={loginPassword} onChange={e => setLoginPassword(e.target.value)}
                                    placeholder="••••••" disabled={isLoading} className={`${inputCls} pr-10`} style={inputStyle} />
                                <button type="button" onClick={() => setShowLoginPw(!showLoginPw)}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 bg-transparent border-none text-txt-muted cursor-pointer hover:text-txt-primary p-0 flex items-center">
                                    {showLoginPw ? <EyeOff size={16} /> : <Eye size={16} />}
                                </button>
                            </div>
                        </div>
                        <button type="submit" disabled={isLoading} className={btnCls}>
                            {isLoading ? <span className="flex items-center justify-center gap-2">
                                <span className="inline-block w-3.5 h-3.5 border-2 border-dark-deeper/30 border-t-dark-deeper rounded-full animate-spin" /> 로그인 중...
                            </span> : '로그인'}
                        </button>
                        <div className="flex justify-center gap-4 mt-4">
                            <button type="button" onClick={() => switchMode('findId')}
                                className="bg-transparent border-none text-txt-muted text-[11px] cursor-pointer hover:text-txt-primary p-0 underline decoration-dark-border hover:decoration-txt-primary">아이디 찾기</button>
                            <span className="text-dark-border text-[11px]">|</span>
                            <button type="button" onClick={() => switchMode('findPw')}
                                className="bg-transparent border-none text-txt-muted text-[11px] cursor-pointer hover:text-txt-primary p-0 underline decoration-dark-border hover:decoration-txt-primary">비밀번호 찾기</button>
                        </div>
                    </form>
                )}

                {/* ───── 회원가입 ───── */}
                {mode === 'register' && (
                    <form onSubmit={handleRegister} className="px-8 pt-5 pb-4">
                        <div className="mb-3">
                            <label className="block text-[11px] text-txt-secondary mb-1.5">아이디 <span className="text-[#f48771]">*</span></label>
                            <input type="text" value={regUsername} onChange={e => setRegUsername(e.target.value)}
                                placeholder="4자 이상" autoFocus disabled={isLoading} className={inputCls} style={inputStyle} />
                        </div>
                        <div className="mb-3">
                            <label className="block text-[11px] text-txt-secondary mb-1.5">비밀번호 <span className="text-[#f48771]">*</span></label>
                            <div className="relative">
                                <input type={showRegPw ? "text" : "password"} value={regPassword} onChange={e => setRegPassword(e.target.value)}
                                    placeholder="6자 이상" disabled={isLoading} className={`${inputCls} pr-10`} style={inputStyle} />
                                <button type="button" onClick={() => setShowRegPw(!showRegPw)}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 bg-transparent border-none text-txt-muted cursor-pointer hover:text-txt-primary p-0 flex items-center">
                                    {showRegPw ? <EyeOff size={16} /> : <Eye size={16} />}
                                </button>
                            </div>
                        </div>
                        <div className="mb-3">
                            <label className="block text-[11px] text-txt-secondary mb-1.5">비밀번호 확인 <span className="text-[#f48771]">*</span></label>
                            <div className="relative">
                                <input type={showRegPwConfirm ? "text" : "password"} value={regPasswordConfirm} onChange={e => setRegPasswordConfirm(e.target.value)}
                                    placeholder="비밀번호 재입력" disabled={isLoading} className={`${inputCls} pr-10`} style={inputStyle} />
                                <button type="button" onClick={() => setShowRegPwConfirm(!showRegPwConfirm)}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 bg-transparent border-none text-txt-muted cursor-pointer hover:text-txt-primary p-0 flex items-center">
                                    {showRegPwConfirm ? <EyeOff size={16} /> : <Eye size={16} />}
                                </button>
                            </div>
                            {regPasswordConfirm && regPassword !== regPasswordConfirm && (
                                <span className="text-[10px] text-[#f48771] mt-1 block">비밀번호가 일치하지 않습니다</span>
                            )}
                        </div>
                        <div className="mb-3">
                            <label className="block text-[11px] text-txt-secondary mb-1.5">이름 <span className="text-[#f48771]">*</span></label>
                            <input type="text" value={regName} onChange={e => setRegName(e.target.value)}
                                placeholder="홍길동" disabled={isLoading} className={inputCls} style={inputStyle} />
                        </div>
                        <div className="mb-3">
                            <label className="block text-[11px] text-txt-secondary mb-1.5">이메일 <span className="text-[#f48771]">*</span></label>
                            <input type="email" value={regEmail} onChange={e => setRegEmail(e.target.value)}
                                placeholder="example@company.com" disabled={isLoading} className={inputCls} style={inputStyle} />
                        </div>
                        <div className="mb-3">
                            <label className="block text-[11px] text-txt-secondary mb-1.5">부서</label>
                            <select value={regDept} onChange={e => setRegDept(e.target.value)}
                                disabled={isLoading} className={selectCls} style={inputStyle}>
                                <option value="">선택하세요</option>
                                {departments.map(d => <option key={d} value={d}>{d}</option>)}
                            </select>
                        </div>
                        <div className="mb-5">
                            <label className="block text-[11px] text-txt-secondary mb-1.5">직책</label>
                            <select value={regRank} onChange={e => setRegRank(e.target.value)}
                                disabled={isLoading} className={selectCls} style={inputStyle}>
                                <option value="">선택하세요</option>
                                {ranks.map(r => <option key={r} value={r}>{r}</option>)}
                            </select>
                        </div>
                        <button type="submit" disabled={isLoading} className={btnCls}>
                            {isLoading ? <span className="flex items-center justify-center gap-2">
                                <span className="inline-block w-3.5 h-3.5 border-2 border-dark-deeper/30 border-t-dark-deeper rounded-full animate-spin" /> 가입 처리 중...
                            </span> : '회원가입'}
                        </button>
                    </form>
                )}

                {/* ───── 아이디 찾기 ───── */}
                {mode === 'findId' && (
                    <form onSubmit={handleFindId} className="px-8 pt-5 pb-4">
                        <p className="text-[12px] text-txt-secondary mb-4">가입 시 등록한 이름과 부서로 아이디를 찾을 수 있습니다.</p>
                        <div className="mb-3">
                            <label className="block text-[11px] text-txt-secondary mb-1.5">이름 <span className="text-[#f48771]">*</span></label>
                            <input type="text" value={findIdName} onChange={e => setFindIdName(e.target.value)}
                                placeholder="홍길동" autoFocus disabled={isLoading} className={inputCls} style={inputStyle} />
                        </div>
                        <div className="mb-5">
                            <label className="block text-[11px] text-txt-secondary mb-1.5">부서 <span className="text-txt-muted">(선택)</span></label>
                            <select value={findIdDept} onChange={e => setFindIdDept(e.target.value)}
                                disabled={isLoading} className={selectCls} style={inputStyle}>
                                <option value="">전체</option>
                                {departments.map(d => <option key={d} value={d}>{d}</option>)}
                            </select>
                        </div>
                        <button type="submit" disabled={isLoading} className={btnCls}>
                            {isLoading ? '조회 중...' : '아이디 찾기'}
                        </button>
                        {foundUsername && (
                            <div className="mt-4 p-3 rounded border border-dark-border bg-dark-light text-center">
                                <p className="text-[11px] text-txt-muted mb-1">찾은 아이디</p>
                                <p className="text-[15px] text-txt-white font-mono font-semibold m-0">{foundUsername}</p>
                            </div>
                        )}
                    </form>
                )}

                {/* ───── 비밀번호 찾기 ───── */}
                {mode === 'findPw' && pwResetStep === 'verify' && (
                    <form onSubmit={handleFindPwVerify} className="px-8 pt-5 pb-4">
                        <p className="text-[12px] text-txt-secondary mb-4">아이디와 이름을 입력하여 본인 확인 후 비밀번호를 재설정합니다.</p>
                        <div className="mb-3">
                            <label className="block text-[11px] text-txt-secondary mb-1.5">아이디 <span className="text-[#f48771]">*</span></label>
                            <input type="text" value={findPwUsername} onChange={e => setFindPwUsername(e.target.value)}
                                placeholder="username" autoFocus disabled={isLoading} className={inputCls} style={inputStyle} />
                        </div>
                        <div className="mb-5">
                            <label className="block text-[11px] text-txt-secondary mb-1.5">이름 <span className="text-[#f48771]">*</span></label>
                            <input type="text" value={findPwName} onChange={e => setFindPwName(e.target.value)}
                                placeholder="홍길동" disabled={isLoading} className={inputCls} style={inputStyle} />
                        </div>
                        <button type="submit" disabled={isLoading} className={btnCls}>
                            {isLoading ? '확인 중...' : '본인 확인'}
                        </button>
                    </form>
                )}

                {mode === 'findPw' && pwResetStep === 'reset' && (
                    <form onSubmit={handleResetPassword} className="px-8 pt-5 pb-4">
                        <div className="mb-3">
                            <label className="block text-[11px] text-txt-secondary mb-1.5">새 비밀번호 <span className="text-[#f48771]">*</span></label>
                            <div className="relative">
                                <input type={showNewPw ? "text" : "password"} value={newPassword} onChange={e => setNewPassword(e.target.value)}
                                    placeholder="6자 이상" autoFocus disabled={isLoading} className={`${inputCls} pr-10`} style={inputStyle} />
                                <button type="button" onClick={() => setShowNewPw(!showNewPw)}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 bg-transparent border-none text-txt-muted cursor-pointer hover:text-txt-primary p-0 flex items-center">
                                    {showNewPw ? <EyeOff size={16} /> : <Eye size={16} />}
                                </button>
                            </div>
                        </div>
                        <div className="mb-5">
                            <label className="block text-[11px] text-txt-secondary mb-1.5">새 비밀번호 확인 <span className="text-[#f48771]">*</span></label>
                            <div className="relative">
                                <input type={showNewPwConfirm ? "text" : "password"} value={newPasswordConfirm} onChange={e => setNewPasswordConfirm(e.target.value)}
                                    placeholder="비밀번호 재입력" disabled={isLoading} className={`${inputCls} pr-10`} style={inputStyle} />
                                <button type="button" onClick={() => setShowNewPwConfirm(!showNewPwConfirm)}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 bg-transparent border-none text-txt-muted cursor-pointer hover:text-txt-primary p-0 flex items-center">
                                    {showNewPwConfirm ? <EyeOff size={16} /> : <Eye size={16} />}
                                </button>
                            </div>
                            {newPasswordConfirm && newPassword !== newPasswordConfirm && (
                                <span className="text-[10px] text-[#f48771] mt-1 block">비밀번호가 일치하지 않습니다</span>
                            )}
                        </div>
                        <button type="submit" disabled={isLoading} className={btnCls}>
                            {isLoading ? '변경 중...' : '비밀번호 변경'}
                        </button>
                    </form>
                )}

                <div className="px-8 pb-5 text-center">
                    <p className="text-[10px] text-txt-muted">© 2026 EasyQuality · GMP Compliance Platform</p>
                </div>
            </div>
        </div>
    )
}
