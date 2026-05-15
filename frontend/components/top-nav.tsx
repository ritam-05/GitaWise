"use client";

import { useEffect, useRef, useState } from "react";
import { Menu } from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";

const menuItems = ["Settings", "Today's Philosophy", "Read Verses", "Disclaimer"];

export function TopNav() {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handlePointerDown(event: MouseEvent) {
      if (!menuRef.current?.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setIsOpen(false);
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
        <div ref={menuRef} className="relative">
          <button
            type="button"
            aria-label="Open options menu"
            aria-expanded={isOpen}
            onClick={() => setIsOpen((current) => !current)}
            className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-transparent text-muted transition-colors hover:border-border hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
          >
            <Menu className="h-5 w-5 stroke-[1.7]" aria-hidden="true" />
          </button>

          {isOpen && (
            <div className="absolute right-0 top-12 z-50 w-52 rounded-2xl border border-border bg-card p-1.5">
              <div className="space-y-1">
                {menuItems.map((item) => (
                  <button
                    key={item}
                    type="button"
                    onClick={() => setIsOpen(false)}
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
