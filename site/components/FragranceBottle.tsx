'use client'
import { motion } from 'framer-motion'

const FAMILY_PALETTE: Record<string, { liquid: string; glow: string; label: string }> = {
  citrus:     { liquid: '#FFD700', glow: '#FFD70055', label: 'Citrus' },
  floral:     { liquid: '#FF69B4', glow: '#FF69B455', label: 'Floral' },
  woody:      { liquid: '#8B4513', glow: '#8B451355', label: 'Woody' },
  oriental:   { liquid: '#FF8C00', glow: '#FF8C0055', label: 'Oriental' },
  musk:       { liquid: '#C8B8A2', glow: '#C8B8A255', label: 'Musk' },
  fresh:      { liquid: '#00CED1', glow: '#00CED155', label: 'Fresh' },
  spicy:      { liquid: '#DC143C', glow: '#DC143C55', label: 'Spicy' },
  gourmand:   { liquid: '#D2691E', glow: '#D2691E55', label: 'Gourmand' },
  leather:    { liquid: '#4A3728', glow: '#4A372855', label: 'Leather' },
  default:    { liquid: '#8B00FF', glow: '#8B00FF55', label: 'Mystery' },
}

interface Props {
  family?: string
  dominance?: number  // 0-1, controls fill level
  animate?: boolean
}

export default function FragranceBottle({ family = 'default', dominance = 0.5, animate = true }: Props) {
  const palette = FAMILY_PALETTE[family] || FAMILY_PALETTE.default
  const fillPct = 30 + dominance * 50  // 30%–80%
  const liquidTop = 200 - (fillPct / 100) * 130

  return (
    <motion.div
      className="relative flex items-center justify-center"
      animate={animate ? { y: [0, -12, 0] } : {}}
      transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
    >
      {/* Glow halo */}
      <div
        className="absolute inset-0 rounded-full blur-3xl opacity-40 pointer-events-none"
        style={{ background: palette.glow }}
      />

      <svg viewBox="0 0 120 240" width={180} height={300} className="relative z-10">
        <defs>
          {/* Bottle glass gradient */}
          <linearGradient id="glass" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="rgba(255,255,255,0.04)" />
            <stop offset="30%"  stopColor="rgba(255,255,255,0.12)" />
            <stop offset="70%"  stopColor="rgba(255,255,255,0.06)" />
            <stop offset="100%" stopColor="rgba(255,255,255,0.02)" />
          </linearGradient>

          {/* Liquid gradient */}
          <linearGradient id="liquid" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%"   stopColor={palette.liquid} stopOpacity="0.9" />
            <stop offset="100%" stopColor={palette.liquid} stopOpacity="0.5" />
          </linearGradient>

          {/* Cap gradient */}
          <linearGradient id="cap" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%"   stopColor="#FFEC8B" />
            <stop offset="100%" stopColor="#B8960C" />
          </linearGradient>

          {/* Shimmer */}
          <linearGradient id="shimmer" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="rgba(255,255,255,0)" />
            <stop offset="50%"  stopColor="rgba(255,255,255,0.18)" />
            <stop offset="100%" stopColor="rgba(255,255,255,0)" />
          </linearGradient>

          {/* Clip for liquid */}
          <clipPath id="bottleClip">
            <path d="M30 85 L22 100 L18 220 Q18 232 30 232 L90 232 Q102 232 102 220 L98 100 L90 85 Z" />
          </clipPath>
        </defs>

        {/* Atomizer pump */}
        <rect x="52" y="10" width="16" height="28" rx="3" fill="url(#cap)" />
        <rect x="57" y="4" width="6" height="10" rx="2" fill="#FFEC8B" />
        <ellipse cx="60" cy="4" rx="4" ry="2.5" fill="#FFD700" />

        {/* Neck */}
        <rect x="44" y="36" width="32" height="52" rx="4"
          fill="rgba(255,255,255,0.06)" stroke="rgba(255,255,255,0.15)" strokeWidth="0.5" />
        {/* Neck shimmer */}
        <rect x="52" y="36" width="8" height="52" fill="rgba(255,255,255,0.06)" />

        {/* Shoulder */}
        <path d="M30 85 Q30 88 26 95 L22 100 L98 100 L94 95 Q90 88 90 85 Z"
          fill="rgba(255,255,255,0.08)" stroke="rgba(255,255,255,0.15)" strokeWidth="0.5" />

        {/* Bottle body outline */}
        <path d="M30 85 L22 100 L18 220 Q18 232 30 232 L90 232 Q102 232 102 220 L98 100 L90 85 Z"
          fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.18)" strokeWidth="0.8" />

        {/* Liquid fill */}
        <g clipPath="url(#bottleClip)">
          <motion.rect
            x="18" width="84" height="240"
            y={liquidTop}
            fill="url(#liquid)"
            initial={{ y: 240 }}
            animate={{ y: liquidTop }}
            transition={{ duration: 1.2, ease: 'easeOut', delay: 0.3 }}
          />
          {/* Wave on liquid surface */}
          <motion.path
            d={`M18 ${liquidTop} Q36 ${liquidTop - 4} 60 ${liquidTop} Q84 ${liquidTop + 4} 102 ${liquidTop} L102 ${liquidTop + 4} L18 ${liquidTop + 4} Z`}
            fill={palette.liquid}
            animate={{ d: [
              `M18 ${liquidTop} Q36 ${liquidTop - 3} 60 ${liquidTop} Q84 ${liquidTop + 3} 102 ${liquidTop}`,
              `M18 ${liquidTop} Q36 ${liquidTop + 3} 60 ${liquidTop} Q84 ${liquidTop - 3} 102 ${liquidTop}`,
            ]}}
            transition={{ duration: 2, repeat: Infinity, repeatType: 'reverse', ease: 'easeInOut' }}
          />
          {/* Liquid shimmer */}
          <motion.rect
            x="18" y={liquidTop} width="84" height="200"
            fill="url(#shimmer)"
            animate={{ x: [18, 60, 18] }}
            transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
          />
        </g>

        {/* Glass reflections */}
        <path d="M28 100 L26 180 Q26 185 30 185 L30 100 Z"
          fill="url(#shimmer)" opacity="0.5" />
        <path d="M92 100 L94 180 Q94 185 90 185 L90 100 Z"
          fill="url(#shimmer)" opacity="0.3" />
        {/* Top glass highlight */}
        <rect x="18" y="100" width="84" height="8"
          fill="url(#glass)" />

        {/* Label area */}
        <rect x="30" y="130" width="60" height="70" rx="3"
          fill="rgba(0,0,0,0.3)" stroke="rgba(255,215,0,0.2)" strokeWidth="0.5" />
        <text x="60" y="152" textAnchor="middle" fontSize="7" fontWeight="600"
          fill="#FFD700" letterSpacing="2" fontFamily="sans-serif">SCENTWAVE</text>
        <line x1="36" y1="156" x2="84" y2="156" stroke="#FFD70044" strokeWidth="0.5" />
        <text x="60" y="167" textAnchor="middle" fontSize="4.5" fill="rgba(255,255,255,0.5)"
          fontFamily="sans-serif" letterSpacing="0.5">AI FRAGRANCE</text>
        <text x="60" y="176" textAnchor="middle" fontSize="4.5" fill="rgba(255,255,255,0.5)"
          fontFamily="sans-serif" letterSpacing="0.5">INTELLIGENCE</text>
        <line x1="36" y1="182" x2="84" y2="182" stroke="#FFD70044" strokeWidth="0.5" />
        <text x="60" y="192" textAnchor="middle" fontSize="5.5" fill={palette.liquid}
          fontFamily="sans-serif" letterSpacing="1">{palette.label.toUpperCase()}</text>

        {/* Base */}
        <rect x="16" y="230" width="88" height="6" rx="2" fill="rgba(255,255,255,0.06)" />
        <rect x="20" y="234" width="80" height="2" rx="1" fill="rgba(255,215,0,0.1)" />
      </svg>
    </motion.div>
  )
}
