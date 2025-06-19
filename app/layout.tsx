import './globals.css'
import { Inter } from 'next/font/google'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'RAG Chat - AI Repository Assistant',
  description: 'Intelligent conversations about GitHub repositories using RAG (Retrieval-Augmented Generation) with Gemini AI',
  keywords: 'RAG, AI, GitHub, repository, chat, assistant, code analysis',
  authors: [{ name: 'Your Name' }],
  viewport: 'width=device-width, initial-scale=1',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} h-full bg-gray-50`}>
        <div className="min-h-full">
          {children}
        </div>
      </body>
    </html>
  )
}
