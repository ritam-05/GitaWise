"use client";

import { useEffect } from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    window.localStorage.setItem("gitawise-theme-v3", "light");
    document.documentElement.classList.remove("dark");
    document.documentElement.classList.add("light");
  }, []);

  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="light"
      enableSystem={false}
      storageKey="gitawise-theme-v3"
      disableTransitionOnChange={false}
    >
      {children}
    </NextThemesProvider>
  );
}
