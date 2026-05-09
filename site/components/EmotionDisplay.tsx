'use client'
import { motion } from 'framer-motion'
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer } from 'recharts'

interface Emotion {
  valence: number
  arousal: number
  dominance: number
}

interface Props { emotion: Emotion }

const LABELS: Record<string, string> = {
  valence: 'Valence',
  arousal: 'Arousal',
  dominance: 'Dominance',
}
const COLORS: Record<string, string> = {
  valence: '#FFD700',
  arousal: '#FF69B4',
  dominance: '#8B00FF',
}
const ICONS: Record<string, string> = {
  valence: '☀',
  arousal: '⚡',
  dominance: '♦',
}
const DESCRIPTIONS: Record<string, (v: number) => string> = {
  valence: v => v > 0.65 ? 'Joyful & bright' : v > 0.40 ? 'Balanced' : 'Melancholic & deep',
  arousal: v => v > 0.65 ? 'Energetic & electric' : v > 0.40 ? 'Moderate' : 'Calm & serene',
  dominance: v => v > 0.65 ? 'Powerful & bold' : v > 0.40 ? 'Grounded' : 'Delicate & soft',
}

export default function EmotionDisplay({ emotion }: Props) {
  const radarData = [
    { axis: 'Valence', value: Math.round(emotion.valence * 100) },
    { axis: 'Arousal', value: Math.round(emotion.arousal * 100) },
    { axis: 'Dominance', value: Math.round(emotion.dominance * 100) },
  ]

  return (
    <div className="space-y-4">
      <h3 className="text-xs font-semibold text-white/40 uppercase tracking-widest">
        Emotional Fingerprint
      </h3>

      {/* Radar */}
      <div className="flex items-center justify-center" style={{ height: 180 }}>
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={radarData} cx="50%" cy="50%" outerRadius={65}>
            <PolarGrid stroke="rgba(255,255,255,0.08)" />
            <PolarAngleAxis
              dataKey="axis"
              tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 11, fontFamily: 'Inter' }}
            />
            <Radar
              dataKey="value"
              stroke="#8B00FF"
              fill="#8B00FF"
              fillOpacity={0.25}
              strokeWidth={2}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Metric bars */}
      <div className="space-y-3">
        {(['valence', 'arousal', 'dominance'] as const).map((key, i) => {
          const val = emotion[key]
          const pct = Math.round(val * 100)
          return (
            <motion.div
              key={key}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
            >
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  <span style={{ color: COLORS[key] }} className="text-sm">{ICONS[key]}</span>
                  <span className="text-white/70 text-xs font-medium uppercase tracking-wider">
                    {LABELS[key]}
                  </span>
                </div>
                <span className="text-xs font-bold" style={{ color: COLORS[key] }}>
                  {pct}%
                </span>
              </div>
              <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: `linear-gradient(90deg, ${COLORS[key]}88, ${COLORS[key]})` }}
                  initial={{ width: '0%' }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 0.8, delay: i * 0.15, ease: 'easeOut' }}
                />
              </div>
              <p className="text-white/25 text-[10px] mt-0.5">{DESCRIPTIONS[key](val)}</p>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}
