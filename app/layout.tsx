import type { Metadata } from 'next';
import { Barlow } from 'next/font/google';
import './globals.css';

const barlow = Barlow({ subsets: ['latin'], weight: ['300', '400', '500', '600'], variable: '--font-barlow' });

export const metadata: Metadata = {
  title: 'Nmatik Labs Mesh Editor',
  description: 'Trim a .ply mesh with X, Y and Z planes and export a closed PLY.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={barlow.variable}>
      <body>{children}</body>
    </html>
  );
}
