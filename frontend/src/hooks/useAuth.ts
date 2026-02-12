import { useState, useEffect, useCallback } from 'react'
import { API_URL } from '../types'

interface User {
    id: number
    username: string
    name: string
    dept: string
    role: string
}

interface AuthState {
    isAuthenticated: boolean
    isLoading: boolean
    user: User | null
}

export function useAuth() {
    const [auth, setAuth] = useState<AuthState>({
        isAuthenticated: false,
        isLoading: true,
        user: null,
    })

    // 앱 시작 시 토큰 검증
    useEffect(() => {
        const token = localStorage.getItem('auth_token')
        if (token) {
            verifyToken(token)
        } else {
            setAuth({ isAuthenticated: false, isLoading: false, user: null })
        }
    }, [])

    const verifyToken = async (token: string) => {
        try {
            const res = await fetch(`${API_URL}/auth/me`, {
                headers: { Authorization: `Bearer ${token}` },
            })
            if (res.ok) {
                const data = await res.json()
                setAuth({ isAuthenticated: true, isLoading: false, user: data.user })
            } else {
                localStorage.removeItem('auth_token')
                setAuth({ isAuthenticated: false, isLoading: false, user: null })
            }
        } catch {
            // 서버 미연결 시에도 토큰이 있으면 임시로 인증 상태 유지
            setAuth({ isAuthenticated: false, isLoading: false, user: null })
        }
    }

    const login = useCallback(async (username: string, password: string) => {
        const res = await fetch(`${API_URL}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        })
        const data = await res.json()
        if (!res.ok) throw new Error(data.detail || '로그인 실패')

        localStorage.setItem('auth_token', data.access_token)
        setAuth({ isAuthenticated: true, isLoading: false, user: data.user })
        return data
    }, [])

    const register = useCallback(async (username: string, password: string, name: string, email: string, dept?: string, rank?: string) => {
        const res = await fetch(`${API_URL}/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password, name, email, dept, rank }),
        })
        const data = await res.json()
        if (!res.ok) throw new Error(data.detail || '회원가입 실패')
        return data
    }, [])

    const logout = useCallback(() => {
        localStorage.removeItem('auth_token')
        setAuth({ isAuthenticated: false, isLoading: false, user: null })
    }, [])

    const changePassword = useCallback(async (currentPassword: string, newPassword: string) => {
        const token = localStorage.getItem('auth_token')
        const res = await fetch(`${API_URL}/auth/password`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
        })
        const data = await res.json()
        if (!res.ok) throw new Error(data.detail || '비밀번호 변경 실패')
        return data
    }, [])

    return {
        ...auth,
        login,
        register,
        logout,
        changePassword,
    }
}
