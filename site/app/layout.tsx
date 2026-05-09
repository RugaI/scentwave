import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'ScentWave — Turn Your Music Into Perfume',
  description: 'AI that transforms any song into a unique fragrance. One song. One real perfume. One formula never made.',
  openGraph: {
    title: 'ScentWave',
    description: 'Turn your music into perfume with AI',
    type: 'website',
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="min-h-screen bg-[#0a0a0a] text-white antialiased">
        {children}
      </body>
    </html>
  )
}
