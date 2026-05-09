'use client'
import { motion } from 'framer-motion'

interface NoteEntry { 0: string; 1: number } // [name, pct]

interface Generated {
  top: [string, number][]
  middle: [string, number][]
  base: [string, number][]
  description: string
  family_profile: Record<string, number>
}

interface Props { generated: Generated; formulaNumber?: number }

const LAYER_CONFIG = [
  { key: 'top' as const,    label: 'Top Notes',    emoji: '🌿', color: '#FFD700', sublabel: 'First impression · evaporates quickly' },
  { key: 'middle' as const, label: 'Heart Notes',  emoji: '🌸', color: '#FF69B4', sublabel: 'The soul · 2–4 hours' },
  { key: 'base' as const,   label: 'Base Notes',   emoji: '🪵', color: '#8B4513', sublabel: 'The foundation · all day' },
]

const FAMILY_COLORS: Record<string, string> = {
  citrus:'#FFD700', floral:'#FF69B4', woody:'#8B4513', oriental:'#FF8C00',
  musk:'#C8B8A2', fresh:'#00CED1', spicy:'#DC143C', gourmand:'#D2691E',
  leather:'#4A3728', aromatic:'#90EE90',
}

export default function FormulaDisplay({ generated, formulaNumber }: Props) {
  const num = formulaNumber ?? Math.floor(Math.random() * 9000 + 1000)
  const dominantFamily = Object.entries(generated.family_profile || {})
    .sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'mystery'

  return (
    <div className="space-y-6">
      {/* Formula header */}
      <motion.div
        initial={{ opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
        className="relative group"
      >
        <div className="absolute -inset-0.5 bg-gradient-to-r from-cyan via-purple to-gold rounded-2xl opacity-25 blur" />
        <div className="relative glass rounded-2xl p-5 text-center border border-cyan/20">
          <p className="text-cyan text-xs tracking-[6px] font-medium uppercase mb-1">
            ✦ Original Formula
          </p>
          <h2 className="text-white font-serif text-2xl font-bold">
            No. <span className="text-gold">{num}</span>
          </h2>
          <p className="text-white/30 text-xs mt-1 italic">A fragrance that has never existed</p>
          <div className="mt-3 flex justify-center gap-2 flex-wrap">
            {Object.entries(generated.family_profile || {})
              .sort((a, b) => b[1] - a[1])
              .slice(0, 4)
              .map(([fam, pct]) => (
                <span
                  key={fam}
                  className="text-[10px] px-2 py-0.5 rounded-full font-medium"
                  style={{
                    background: (FAMILY_COLORS[fam] || '#8B00FF') + '22',
                    color: FAMILY_COLORS[fam] || '#8B00FF',
                    border: `1px solid ${(FAMILY_COLORS[fam] || '#8B00FF')}44`,
                  }}
                >
                  {fam} {pct.toFixed(0)}%
                </span>
              ))}
          </div>
        </div>
      </motion.div>

      {/* Pyramid layers */}
      {LAYER_CONFIG.map(({ key, label, emoji, color, sublabel }, layerIdx) => {
        const notes = generated[key] ?? []
        return (
          <motion.div
            key={key}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: layerIdx * 0.15 }}
            className="space-y-2"
          >
            <div className="flex items-baseline gap-2">
              <span className="text-sm">{emoji}</span>
              <span className="text-xs font-semibold uppercase tracking-widest" style={{ color }}>
                {label}
              </span>
              <span className="text-white/20 text-[10px]">{sublabel}</span>
            </div>
            <div className="space-y-1.5">
              {notes.slice(0, 5).map(([name, pct], j) => (
                <motion.div
                  key={name}
                  initial={{ width: 0, opacity: 0 }}
                  animate={{ width: '100%', opacity: 1 }}
                  transition={{ delay: layerIdx * 0.15 + j * 0.05 }}
                  className="flex items-center gap-3"
                >
                  <span className="text-white/60 text-xs w-32 truncate shrink-0">
                    {name}
                  </span>
                  <div className="flex-1 h-1 bg-white/5 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ background: `linear-gradient(90deg, ${color}66, ${color})` }}
                      initial={{ width: '0%' }}
                      animate={{ width: `${pct}%` }}
                      transition={{ duration: 0.7, delay: layerIdx * 0.15 + j * 0.06, ease: 'easeOut' }}
                    />
                  </div>
                  <span className="text-[10px] font-bold shrink-0" style={{ color }}>
                    {pct.toFixed(0)}%
                  </span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )
      })}

      {/* Luxury description */}
      {generated.description && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="relative"
        >
          <div className="absolute left-0 top-0 bottom-0 w-0.5 bg-gradient-to-b from-purple via-cyan to-transparent rounded" />
          <div className="pl-4">
            <p className="text-xs text-white/30 uppercase tracking-widest mb-2">Description</p>
            <p className="text-white/60 text-sm leading-relaxed italic font-serif">
              {generated.description}
            </p>
          </div>
        </motion.div>
      )}
    </div>
  )
}
