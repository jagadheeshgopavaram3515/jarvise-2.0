import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = { title: 'JARVIS | Personal Assistant', description: 'JARVIS voice assistant and live system report' };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
