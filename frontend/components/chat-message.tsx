 "use client";

import { motion } from "framer-motion";
import { CitationCard } from "@/components/citation-card";
import { cn } from "@/lib/utils";

export type Citation = {
  chapter: number;
  verse: number;
  shloka: string;
  meaning: string;
  tags: string[];
};

type ChatMessageProps = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
};

export function ChatMessage({ role, content, citations = [] }: ChatMessageProps) {
  const isUser = role === "user";

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28, ease: "easeOut" }}
      className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}
    >
      <div
        className={cn(
          isUser
            ? "max-w-[60%] rounded-2xl bg-surface/60 px-4 py-2.5"
            : "w-full space-y-3",
        )}
      >
        <div
          className={cn(
            "whitespace-pre-wrap",
            isUser
              ? "text-[15px] leading-6 text-foreground"
              : "text-[16px] leading-[1.65] text-foreground",
          )}
        >
          {content}
        </div>
        {!isUser && citations.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {citations.map((citation) => (
              <CitationCard
                key={`${citation.chapter}-${citation.verse}`}
                {...citation}
              />
            ))}
          </div>
        )}
      </div>
    </motion.section>
  );
}
