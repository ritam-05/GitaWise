"use client";

import { useEffect, useRef, useState } from "react";
import { Menu, UserRound } from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";

const menuItems = ["Settings", "Today's Philosophy", "Read Verses", "Disclaimer"];

export function TopNav() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const profileRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handlePointerDown(event: MouseEvent) {
      const target = event.target as Node;

      if (!menuRef.current?.contains(target)) {
        setIsMenuOpen(false);
      }

      if (!profileRef.current?.contains(target)) {
        setIsProfileOpen(false);
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setIsMenuOpen(false);
        setIsProfileOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  return (
    <header className="flex w-full items-center justify-end px-4 py-4 sm:px-6">
      <div className="flex items-center gap-2">
        <ThemeToggle />
        <div ref={profileRef} className="relative">
          <button
            type="button"
            aria-label="Open profile menu"
            aria-expanded={isProfileOpen}
            onClick={() => {
              setIsProfileOpen((current) => !current);
              setIsMenuOpen(false);
            }}
            className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-transparent text-muted transition-colors hover:border-border hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
          >
            <UserRound className="h-[18px] w-[18px] stroke-[1.7]" aria-hidden="true" />
          </button>

          {isProfileOpen && (
            <div className="absolute right-0 top-12 z-50 w-56 rounded-2xl border border-border bg-card p-1.5">
              <div className="border-b border-border px-3 py-3">
                <p className="text-sm font-medium text-foreground">Guest</p>
                <p className="mt-1 text-xs text-muted">
                  Sign in to personalize GitaWise
                </p>
              </div>
              <div className="space-y-1 pt-1.5">
                <button
                  type="button"
                  onClick={() => setIsProfileOpen(false)}
                  className="w-full rounded-xl px-3 py-2.5 text-left text-sm text-secondary transition-colors hover:bg-surface hover:text-foreground"
                >
                  Sign In
                </button>
                <button
                  type="button"
                  onClick={() => setIsProfileOpen(false)}
                  className="w-full rounded-xl px-3 py-2.5 text-left text-sm text-secondary transition-colors hover:bg-surface hover:text-foreground"
                >
                  Account Settings
                </button>
              </div>
            </div>
          )}
        </div>
        <div ref={menuRef} className="relative">
          <button
            type="button"
            aria-label="Open options menu"
            aria-expanded={isMenuOpen}
            onClick={() => {
              setIsMenuOpen((current) => !current);
              setIsProfileOpen(false);
            }}
            className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-transparent text-muted transition-colors hover:border-border hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
          >
            <Menu className="h-5 w-5 stroke-[1.7]" aria-hidden="true" />
          </button>

          {isMenuOpen && (
            <div className="absolute right-0 top-12 z-50 w-52 rounded-2xl border border-border bg-card p-1.5">
              <div className="space-y-1">
                {menuItems.map((item) => (
                  <button
                    key={item}
                    type="button"
                    onClick={() => setIsMenuOpen(false)}
                    className="w-full rounded-xl px-3 py-2.5 text-left text-sm text-secondary transition-colors hover:bg-surface hover:text-foreground"
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
