'use client'
import { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, Loader2, ChevronDown, Sparkles, Github } from 'lucide-react'
import FragranceBottle from '@/components/FragranceBottle'
import SpotifyPlayer from '@/components/SpotifyPlayer'
import EmotionDisplay from '@/components/EmotionDisplay'
import PerfumeMatches from '@/components/PerfumeMatches'
import FormulaDisplay from '@/components/FormulaDisplay'

// ── Types ────────────────────────────────────────────────────────────────────
interface SongInfo { name: string; artist: string; track_id?: string; spotify_url?: string; embed_url?: string }
interface Emotion { valence: number; arousal: number; dominance: number }
interface AnalysisResult {
  song: SongInfo
  emotion: Emotion
  retrieved: any[]
  generated: any
}
interface SearchResult { track_id: string | null; name: string; artist: string; genre: string }

// ── Quick presets ─────────────────────────────────────────────────────────────
const PRESETS = [
  { genre: '🎼 Classical', songs: ['Clair de Lune', 'Moonlight Sonata', 'Four Seasons Spring', 'Canon in D', 'Symphony No. 9 Beethoven'] },
  { genre: '🎷 Jazz', songs: ['Take Five', 'So What', 'Autumn Leaves', "Summertime Gershwin", 'My Favorite Things'] },
  { genre: '🎸 Rock', songs: ['Bohemian Rhapsody', 'Smells Like Teen Spirit', 'Sweet Child O Mine', 'Black', 'Hotel California'] },
  { genre: '🎤 Hip-Hop', songs: ['Lose Yourself', 'HUMBLE.', 'N.Y. State of Mind', 'Electric Relaxation'] },
  { genre: '🌊 Ambient', songs: ['Weightless', 'Strobe', 'Experience', 'An Ending Ascent'] },
  { genre: '💃 Pop', songs: ['Blinding Lights', 'Bad Guy', 'Shape of You', 'Stay'] },
  { genre: '🎻 Soul', songs: ['A Change Is Gonna Come', 'Respect', 'I Want You Back', 'What s Going On'] },
]

const FAMILY_FROM_PROFILE = (fp: Record<string, number>) =>
  Object.entries(fp || {}).sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'default'

// ── Background particles ──────────────────────────────────────────────────────
function Particles() {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
      {Array.from({ length: 24 }).map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-px h-px rounded-full opacity-30"
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
            background: ['#FFD700', '#8B00FF', '#00CED1', '#FF69B4'][i % 4],
            width: Math.random() * 3 + 1,
            height: Math.random() * 3 + 1,
          }}
          animate={{
            y: [0, -Math.random() * 120 - 40],
            opacity: [0, 0.4, 0],
          }}
          transition={{
            duration: Math.random() * 8 + 6,
            repeat: Infinity,
            delay: Math.random() * 8,
            ease: 'easeInOut',
          }}
        />
      ))}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function Home() {
  const [song, setSong]         = useState('')
  const [artist, setArtist]     = useState('')
  const [loading, setLoading]   = useState(false)
  const [result, setResult]     = useState<AnalysisResult | null>(null)
  const [error, setError]       = useState('')
  const [suggestions, setSugs]  = useState<SearchResult[]>([])
  const [showSugs, setShowSugs] = useState(false)
  const [formulaNum]            = useState(() => Math.floor(Math.random() * 9000 + 1000))
  const resultsRef              = useRef<HTMLDivElement>(null)
  const searchRef               = useRef<HTMLDivElement>(null)
  const debounceRef             = useRef<NodeJS.Timeout | null>(null)

  // Live search suggestions
  const fetchSuggestions = useCallback(async (q: string) => {
    if (q.length < 2) { setSugs([]); return }
    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`)
      const data = await res.json()
      setSugs(data)
      setShowSugs(true)
    } catch { setSugs([]) }
  }, [])

  const onSongChange = (val: string) => {
    setSong(val)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => fetchSuggestions(val), 300)
  }

  const selectSuggestion = (s: SearchResult) => {
    setSong(s.name)
    setArtist(s.artist)
    setShowSugs(false)
    setSugs([])
  }

  // Close suggestions on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setShowSugs(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const analyze = async (songName = song, artistName = artist) => {
    if (!songName.trim()) return
    setLoading(true)
    setError('')
    setShowSugs(false)
    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ song: songName, artist: artistName }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Analysis failed')
      setResult(data)
      // Smooth scroll to results
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100)
    } catch (e: any) {
      setError(e.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  const dominantFamily = result ? FAMILY_FROM_PROFILE(result.generated?.family_profile) : 'default'

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white relative">
      <Particles />

      {/* ── Nav ── */}
      <nav className="fixed top-0 w-full z-50 border-b border-white/5 backdrop-blur-xl bg-[#0a0a0a]/80">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-gold font-serif font-bold text-xl tracking-widest">SCENT</span>
            <span className="text-purple font-serif font-bold text-xl tracking-widest">WAVE</span>
          </div>
          <div className="flex items-center gap-6">
            <a href="#discover" className="text-white/40 hover:text-white text-sm transition-colors">Discover</a>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-white/40 hover:text-white transition-colors"
            >
              <Github size={18} />
            </a>
          </div>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section id="discover" className="min-h-screen flex items-center justify-center pt-16 px-6 relative">
        {/* Background gradient */}
        <div className="absolute inset-0 bg-gradient-radial from-purple/5 via-transparent to-transparent pointer-events-none" />
        <div className="absolute top-1/3 left-1/4 w-96 h-96 bg-purple/5 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute top-1/2 right-1/4 w-64 h-64 bg-cyan/5 rounded-full blur-3xl pointer-events-none" />

        <div className="max-w-5xl w-full grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          {/* Left: Bottle + title */}
          <div className="flex flex-col items-center lg:items-start gap-8">
            <div className="flex justify-center">
              <FragranceBottle
                family={result ? dominantFamily : 'default'}
                dominance={result ? result.emotion.dominance : 0.5}
                animate={true}
              />
            </div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-center lg:text-left"
            >
              <p className="text-xs tracking-[6px] text-purple/80 uppercase mb-3">
                ✦ AI FRAGRANCE INTELLIGENCE ✦
              </p>
              <h1 className="text-4xl lg:text-5xl font-serif font-bold leading-tight">
                <span className="text-shimmer">Your Music</span>
                <br />
                <span className="text-white">Has a Scent</span>
              </h1>
              <p className="text-white/40 text-sm mt-4 leading-relaxed max-w-sm">
                One song → one real perfume match → one formula that has never been made.
                <br />Powered by 2,465 fragrances and 89,741 songs.
              </p>
            </motion.div>
          </div>

          {/* Right: Search */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="space-y-6"
          >
            {/* Search card */}
            <div className="relative group">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-purple via-cyan to-gold rounded-2xl opacity-20 group-hover:opacity-40 blur transition duration-500" />
              <div className="relative glass rounded-2xl p-6 space-y-4" ref={searchRef}>
                <p className="text-white/50 text-xs uppercase tracking-widest">Enter a song</p>

                {/* Song input with suggestions */}
                <div className="relative">
                  <div className="flex items-center gap-3 bg-white/5 border border-white/10 rounded-xl px-4 py-3
                                  focus-within:border-purple/60 transition-all">
                    <Search size={16} className="text-white/30 shrink-0" />
                    <input
                      className="search-input flex-1"
                      placeholder="Song name..."
                      value={song}
                      onChange={e => onSongChange(e.target.value)}
                      onFocus={() => song.length >= 2 && setShowSugs(true)}
                      onKeyDown={e => e.key === 'Enter' && analyze()}
                    />
                  </div>

                  {/* Suggestions dropdown */}
                  <AnimatePresence>
                    {showSugs && suggestions.length > 0 && (
                      <motion.div
                        initial={{ opacity: 0, y: -8 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -8 }}
                        className="absolute top-full left-0 right-0 mt-2 glass border border-white/10 rounded-xl overflow-hidden z-50 shadow-2xl"
                      >
                        {suggestions.map((s, i) => (
                          <button
                            key={i}
                            onClick={() => selectSuggestion(s)}
                            className="w-full flex items-center gap-3 px-4 py-3 hover:bg-white/5 transition-colors text-left"
                          >
                            <div className="w-8 h-8 rounded-lg bg-purple/20 flex items-center justify-center shrink-0">
                              <span className="text-xs">🎵</span>
                            </div>
                            <div className="min-w-0">
                              <p className="text-white text-sm truncate">{s.name}</p>
                              <p className="text-white/30 text-xs truncate">{s.artist}{s.genre ? ` · ${s.genre}` : ''}</p>
                            </div>
                          </button>
                        ))}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* Artist input */}
                <div className="flex items-center gap-3 bg-white/5 border border-white/10 rounded-xl px-4 py-3
                                focus-within:border-purple/60 transition-all">
                  <span className="text-white/30 text-sm shrink-0">by</span>
                  <input
                    className="search-input"
                    placeholder="Artist (optional)"
                    value={artist}
                    onChange={e => setArtist(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && analyze()}
                  />
                </div>

                {/* Submit */}
                <button
                  onClick={() => analyze()}
                  disabled={loading || !song.trim()}
                  className="w-full py-3.5 rounded-xl font-semibold text-sm tracking-wide
                             bg-gradient-to-r from-purple via-purple-light to-cyan
                             hover:opacity-90 active:scale-[0.98] transition-all
                             disabled:opacity-40 disabled:cursor-not-allowed
                             flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <><Loader2 size={16} className="animate-spin" /> Analyzing...</>
                  ) : (
                    <><Sparkles size={16} /> Discover My Scent</>
                  )}
                </button>

                {error && (
                  <p className="text-red-400/80 text-xs text-center">{error}</p>
                )}
              </div>
            </div>

            {/* Quick presets */}
            <div>
              <p className="text-white/25 text-xs uppercase tracking-widest mb-3">Quick picks</p>
              <div className="space-y-2">
                {PRESETS.map(({ genre, songs }) => (
                  <details key={genre} className="group">
                    <summary className="cursor-pointer text-xs text-white/40 hover:text-white/70
                                       transition-colors list-none flex items-center gap-2 py-1">
                      <ChevronDown size={12} className="transition-transform group-open:rotate-180" />
                      {genre}
                    </summary>
                    <div className="pl-5 pt-2 flex flex-wrap gap-1.5">
                      {songs.map(s => (
                        <button
                          key={s}
                          onClick={() => { setSong(s); analyze(s, '') }}
                          className="text-[11px] px-2.5 py-1 rounded-full bg-white/5 hover:bg-purple/20
                                     border border-white/10 hover:border-purple/40 text-white/50
                                     hover:text-white transition-all"
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  </details>
                ))}
              </div>
            </div>
          </motion.div>
        </div>

        {/* Scroll hint */}
        {!result && (
          <motion.div
            className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-white/20"
            animate={{ y: [0, 6, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <span className="text-[10px] uppercase tracking-widest">Scroll</span>
            <ChevronDown size={14} />
          </motion.div>
        )}
      </section>

      {/* ── Results ── */}
      <AnimatePresence>
        {result && (
          <motion.section
            ref={resultsRef}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="px-6 pb-24 pt-12 max-w-6xl mx-auto"
          >
            {/* Song header */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center mb-12"
            >
              <p className="text-white/30 text-xs uppercase tracking-widest mb-2">Analysis complete</p>
              <h2 className="text-3xl font-serif font-bold text-shimmer">
                {result.song.name}
              </h2>
              {result.song.artist && (
                <p className="text-white/40 mt-1">{result.song.artist}</p>
              )}
              <div className="flex justify-center mt-4">
                <div className="h-px w-32 bg-gradient-to-r from-transparent via-gold/50 to-transparent" />
              </div>
            </motion.div>

            {/* 3-column layout */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

              {/* Left column: Spotify + Emotion + Bottle */}
              <div className="space-y-8">
                {/* Spotify embed */}
                <SpotifyPlayer
                  trackId={result.song.track_id}
                  songName={result.song.name}
                  artist={result.song.artist}
                  spotifyUrl={result.song.spotify_url}
                />

                {/* Emotion */}
                <div className="glass rounded-2xl p-5">
                  <EmotionDisplay emotion={result.emotion} />
                </div>
              </div>

              {/* Middle column: Perfume Matches + small bottle */}
              <div className="space-y-6">
                {/* Animated bottle for result */}
                <div className="flex justify-center">
                  <FragranceBottle
                    family={dominantFamily}
                    dominance={result.emotion.dominance}
                    animate={true}
                  />
                </div>

                <div className="glass rounded-2xl p-5">
                  <PerfumeMatches matches={result.retrieved} />
                </div>
              </div>

              {/* Right column: Novel Formula */}
              <div>
                <div className="glass rounded-2xl p-5">
                  <FormulaDisplay generated={result.generated} formulaNumber={formulaNum} />
                </div>
              </div>
            </div>

            {/* Analyze another */}
            <div className="text-center mt-16">
              <button
                onClick={() => {
                  setResult(null)
                  setSong('')
                  setArtist('')
                  window.scrollTo({ top: 0, behavior: 'smooth' })
                }}
                className="text-sm text-white/30 hover:text-gold transition-colors underline underline-offset-4"
              >
                ↑ Analyze another song
              </button>
            </div>
          </motion.section>
        )}
      </AnimatePresence>

      {/* ── Footer ── */}
      <footer className="border-t border-white/5 py-8 text-center text-white/20 text-xs">
        <p>
          <span className="text-gold font-serif">SCENTWAVE</span>
          {' '}· AI Fragrance Intelligence · 2,465 fragrances · 89,741 songs
        </p>
        <p className="mt-1">Built with PyTorch · Next.js · Gradio</p>
      </footer>
    </div>
  )
}
