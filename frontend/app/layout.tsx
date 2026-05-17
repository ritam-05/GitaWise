import type { Metadata } from "next";
import { GeistMono } from "geist/font/mono";
import { ThemeProvider } from "@/components/theme-provider";
import { TopNav } from "@/components/top-nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "GitaWise",
  description: "A calm philosophical AI companion for Bhagavad Gita reflection.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="light" suppressHydrationWarning>
      <body
        className={`${GeistMono.variable} min-h-screen bg-background font-sans text-foreground antialiased`}
      >
        <ThemeProvider>
          <div className="flex min-h-screen flex-col">
            <TopNav />
            <main className="flex-1">{children}</main>
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
