"use client";

import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";
import { useEffect, useState } from "react";

type Props = {
  isOpen: boolean;
  onClose: () => void;
};

export function TodaysPhilosophyModal({ isOpen, onClose }: Props) {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<null | {
    summary: string;
    shloka: string;
    chapter: number | null;
    verse: number | null;
    citation: string;
  }>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    let mounted = true;
    setLoading(true);
    setError(null);

    const backendUrl = (process.env.BACKEND_URL || "http://127.0.0.1:8000") as string;
    fetch(`${backendUrl}/query-engine/today-philosophy`)
      .then(async (res) => {
        if (!res.ok) {
          const t = await res.text();
          throw new Error(t || "Failed to fetch today's philosophy");
        }
        return res.json();
      })
      .then((json) => {
        if (!mounted) return;
        setData(json);
      })
      .catch((err) => {
        if (!mounted) return;
        setError(String(err.message || err));
      })
      .finally(() => mounted && setLoading(false));

    return () => {
      mounted = false;
    };
  }, [isOpen]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.6 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
            className="fixed inset-0 z-40 bg-black"
          />

          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
          >
            <div className="relative max-h-[80vh] w-full max-w-2xl overflow-y-auto rounded-2xl bg-card p-6 shadow-[0_20px_60px_rgba(0,0,0,0.3)]">
              <button
                type="button"
                onClick={onClose}
                className="absolute left-4 top-4 inline-flex h-8 w-8 items-center justify-center rounded-full border border-transparent text-muted transition-colors hover:border-border hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
                aria-label="Close"
              >
                <X className="h-5 w-5" aria-hidden="true" />
              </button>

              <div className="pr-12">
                <h2 className="text-center text-2xl font-semibold text-foreground mb-4">
                  Today's Philosophy
                </h2>

                <div className="space-y-4 text-[15px] leading-7 text-secondary">
                  {loading && <p>Loading…</p>}
                  {error && <p className="text-destructive">{error}</p>}

                  {!loading && !error && data && (
                    <>
                      <div className="prose max-w-none text-foreground">
                        <p>{data.summary || data.shloka}</p>
                      </div>

                      <p className="mt-4 text-sm text-muted">
                        {data.citation}
                      </p>
                    </>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
