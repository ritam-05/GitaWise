"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

type CitationCardProps = {
  chapter: number;
  verse: number;
  shloka: string;
  meaning: string;
  tags: string[];
  summary?: string;
};

export function CitationCard({
  chapter,
  verse,
  shloka,
  meaning,
  tags,
  summary,
}: CitationCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="inline-block align-middle">
      <button
        type="button"
        onClick={() => setExpanded((value) => !value)}
        className="inline-flex items-center gap-1.5 rounded-full bg-surface/50 px-2.5 py-1 text-[14px] font-medium text-secondary transition hover:bg-surface hover:text-foreground"
      >
        <span>Gita {chapter}.{verse}</span>
        {expanded && (
          <motion.svg
            className="h-3.5 w-3.5"
            initial={{ rotate: 0 }}
            animate={{ rotate: 180 }}
            transition={{ duration: 0.2 }}
            viewBox="0 0 12 12"
          >
            <path
              stroke="currentColor"
              strokeWidth="2"
              d="M6 9l-3-3-3 3"
              fill="none"
              strokeLinecap="round"
            />
          </motion.svg>
        )}
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="overflow-hidden"
          >
            <div className="mt-1.5 rounded-2xl bg-surface/60 p-3 shadow-[0_4px_12px_rgba(0,0,0,0.08)]">
              <blockquote className="text-[14px] leading-6 italic text-secondary">
                {shloka}
              </blockquote>
              {summary && (
                <p className="mt-2 text-[13px] leading-5 text-secondary">{summary}</p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
