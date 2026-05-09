'use client'
import { motion } from 'framer-motion'
import { Music, ExternalLink } from 'lucide-react'

interface Props {
  trackId?: string | null
  songName: string
  artist: string
  spotifyUrl?: string | null
}

export default function SpotifyPlayer({ trackId, songName, artist, spotifyUrl }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="w-full"
    >
      {trackId ? (
        <div className="relative group spotify-embed">
          {/* Glow ring */}
          <div className="absolute -inset-0.5 bg-gradient-to-r from-purple to-cyan rounded-[14px] opacity-30 group-hover:opacity-60 blur transition duration-500" />
          <div className="relative rounded-[14px] overflow-hidden bg-[#111]">
            <iframe
              src={`https://open.spotify.com/embed/track/${trackId}?utm_source=generator&theme=0`}
              width="100%"
              height="352"
              frameBorder="0"
              allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
              loading="lazy"
              className="block"
            />
          </div>
          {/* Spotify link */}
          {spotifyUrl && (
            <a
              href={spotifyUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-3 flex items-center gap-2 text-xs text-white/40 hover:text-[#1DB954] transition-colors"
            >
              <ExternalLink size={12} />
              Open in Spotify
            </a>
          )}
        </div>
      ) : (
        /* Fallback: no track_id */
        <div className="relative group">
          <div className="absolute -inset-0.5 bg-gradient-to-r from-purple to-cyan rounded-2xl opacity-20 blur" />
          <div className="relative glass rounded-2xl p-8 flex flex-col items-center gap-4 text-center">
            <div className="w-20 h-20 rounded-full bg-[#1DB954]/10 flex items-center justify-center">
              <Music size={36} className="text-[#1DB954]" />
            </div>
            <div>
              <p className="text-white font-semibold text-lg font-serif">{songName}</p>
              <p className="text-white/50 text-sm mt-1">{artist || 'Unknown Artist'}</p>
            </div>
            <p className="text-white/30 text-xs">
              Spotify embed unavailable for this track.
            </p>
            {/* Search on Spotify */}
            <a
              href={`https://open.spotify.com/search/${encodeURIComponent(`${songName} ${artist}`)}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-xs text-[#1DB954]/70 hover:text-[#1DB954] transition-colors"
            >
              <ExternalLink size={12} />
              Search on Spotify
            </a>
          </div>
        </div>
      )}
    </motion.div>
  )
}
