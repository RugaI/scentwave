import { NextRequest, NextResponse } from 'next/server'

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

export async function GET(req: NextRequest) {
  const q = req.nextUrl.searchParams.get('q') || ''
  try {
    const res = await fetch(`${BACKEND}/search?q=${encodeURIComponent(q)}`)
    const data = await res.json()
    return NextResponse.json(data)
  } catch {
    return NextResponse.json([], { status: 503 })
  }
}
