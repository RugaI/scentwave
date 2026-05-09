'use client'
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, Award } from 'lucide-react'

const MEDALS = ['🥇', '🥈', '🥉']
const FAMILY_COLORS: Record<string, string> = {
  citrus: '#FFD700', floral: '#FF69B4', woody: '#8B4513',
  oriental: '#FF8C00', musk: '#C8B8A2', fresh: '#00CED1',
  spicy: '#DC143C', gourmand: '#D2691E', leather: '#4A3728',
}

interface Match {
  rank: number
  name: string
  brand: string
  match_pct: number
  family: string
  top_notes: string
  middle_notes: string
  base_notes: string
  mood_tags: string
}

interface Props { matches: Match[] }

export default function PerfumeMatches({ matches }: Props) {
  const [expanded, setExpanded] = useState<number | null>(0)

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold text-white/40 uppercase tracking-widest">
        Closest Existing Perfumes
      </h3>

      {matches.map((m, i) => {
        const familyKey = (m.family || '').split('_')[0]
        const color = FAMILY_COLORS[familyKey] || '#8B00FF'
        const isOpen = expanded === i

        return (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.12 }}
            className="result-card glass rounded-xl overflow-hidden border border-white/5"
            style={{ borderLeftColor: color + '66', borderLeftWidth: 3 }}
          >
            {/* Header */}
            <button
              onClick={() => setExpanded(isOpen ? null : i)}
              className="w-full flex items-center justify-between p-4 text-left"
            >
              <div className="flex items-center gap-3">
                <span className="text-xl">{MEDALS[i]}</span>
                <div>
                  <p className="font-semibold text-white font-serif" style={{ color: i === 0 ? '#FFD700' : 'white' }}>
                    {m.name}
                  </p>
                  <p className="text-white/40 text-xs">{m.brand}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <span
                  className="text-xs font-bold px-2.5 py-1 rounded-full"
                  style={{ background: color + '22', color, border: `1px solid ${color}44` }}
                >
                  {m.match_pct.toFixed(0)}%
                </span>
                <motion.div animate={{ rotate: isOpen ? 180 : 0 }} transition={{ duration: 0.2 }}>
                  <ChevronDown size={16} className="text-white/30" />
                </motion.div>
              </div>
            </button>

            {/* Expanded notes */}
            <AnimatePresence>
              {isOpen && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.25 }}
                  className="overflow-hidden"
                >
                  <div className="px-4 pb-4 space-y-3 border-t border-white/5 pt-3">
                    {/* Family badge */}
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] uppercase tracking-widest text-white/30">Family</span>
                      <span
                        className="text-[10px] px-2 py-0.5 rounded-full font-medium"
                        style={{ background: color + '22', color }}
                      >
                        {m.family?.replace(/_/g, ' ')}
                      </span>
                    </div>

                    {/* Notes pyramid */}
                    {[
                      { label: 'Top Notes', notes: m.top_notes, emoji: '🌿' },
                      { label: 'Heart Notes', notes: m.middle_notes, emoji: '🌸' },
                      { label: 'Base Notes', notes: m.base_notes, emoji: '🪵' },
                    ].map(({ label, notes, emoji }) => notes && (
                      <div key={label}>
                        <p className="text-[10px] uppercase tracking-widest text-white/25 mb-1.5">
                          {emoji} {label}
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                          {notes.split(/[|,]/).filter(Boolean).map(n => (
                            <span key={n} className="note-pill text-white/60">
                              {n.trim().replace(/_/g, ' ')}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}

                    {/* Mood tags */}
                    {m.mood_tags && (
                      <div className="flex flex-wrap gap-1.5">
                        {m.mood_tags.split(/[|,]/).filter(Boolean).slice(0, 4).map(t => (
                          <span key={t} className="text-[10px] text-white/20 italic">#{t.trim()}</span>
                        ))}
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )
      })}
    </div>
  )
}
