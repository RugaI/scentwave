import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        gold: '#FFD700',
        'gold-dim': '#B8960C',
        purple: {
          DEFAULT: '#8B00FF',
          light: '#A855F7',
          dim: '#4B0082',
        },
        cyan: '#00CED1',
        pink: '#FF69B4',
        woody: '#8B4513',
        dark: {
          DEFAULT: '#0a0a0a',
          card: '#111111',
          border: '#1e1e1e',
        },
      },
      fontFamily: {
        serif: ['Playfair Display', 'Georgia', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        float: 'float 4s ease-in-out infinite',
        shimmer: 'shimmer 2.5s linear infinite',
        pulse_slow: 'pulse 3s ease-in-out infinite',
        'spin-slow': 'spin 8s linear infinite',
        glow: 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-14px)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        glow: {
          '0%': { filter: 'drop-shadow(0 0 8px currentColor)' },
          '100%': { filter: 'drop-shadow(0 0 24px currentColor)' },
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
      },
    },
  },
  plugins: [],
}
export default config
