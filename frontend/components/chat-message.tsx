 "use client";

import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CitationCard } from "@/components/citation-card";
import { cn } from "@/lib/utils";

export type Citation = {
  chapter: number;
  verse: number;
  shloka: string;
  meaning: string;
  tags: string[];
  summary?: string;
};

type ChatMessageProps = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
};

export function ChatMessage({ role, content, citations = [] }: ChatMessageProps) {
  const isUser = role === "user";

  const markdownComponents = {
    h1: (props: React.ComponentPropsWithoutRef<"h1">) => (
      <h1 {...props} className={cn("mt-2 text-lg font-semibold tracking-tight", props.className)} />
    ),
    h2: (props: React.ComponentPropsWithoutRef<"h2">) => (
      <h2 {...props} className={cn("mt-2 text-base font-semibold tracking-tight", props.className)} />
    ),
    h3: (props: React.ComponentPropsWithoutRef<"h3">) => (
      <h3 {...props} className={cn("mt-2 text-sm font-semibold tracking-tight", props.className)} />
    ),
    p: (props: React.ComponentPropsWithoutRef<"p">) => (
      <p {...props} className={cn("mb-3 last:mb-0", props.className)} />
    ),
    ul: (props: React.ComponentPropsWithoutRef<"ul">) => (
      <ul {...props} className={cn("mb-3 list-disc space-y-1 pl-5 last:mb-0", props.className)} />
    ),
    ol: (props: React.ComponentPropsWithoutRef<"ol">) => (
      <ol {...props} className={cn("mb-3 list-decimal space-y-1 pl-5 last:mb-0", props.className)} />
    ),
    li: (props: React.ComponentPropsWithoutRef<"li">) => (
      <li {...props} className={cn("leading-6", props.className)} />
    ),
    strong: (props: React.ComponentPropsWithoutRef<"strong">) => (
      <strong {...props} className={cn("font-semibold text-foreground", props.className)} />
    ),
    em: (props: React.ComponentPropsWithoutRef<"em">) => (
      <em {...props} className={cn("italic text-foreground/90", props.className)} />
    ),
    blockquote: (props: React.ComponentPropsWithoutRef<"blockquote">) => (
      <blockquote
        {...props}
        className={cn(
          "mb-3 border-l-2 border-border pl-4 italic text-foreground/85 last:mb-0",
          props.className,
        )}
      />
    ),
    table: (props: React.ComponentPropsWithoutRef<"table">) => (
      <div className="mb-3 overflow-x-auto last:mb-0">
        <table {...props} className={cn("w-full border-collapse text-left text-sm", props.className)} />
      </div>
    ),
    thead: (props: React.ComponentPropsWithoutRef<"thead">) => (
      <thead {...props} className={cn("border-b border-border", props.className)} />
    ),
    th: (props: React.ComponentPropsWithoutRef<"th">) => (
      <th {...props} className={cn("px-2 py-1 font-semibold", props.className)} />
    ),
    td: (props: React.ComponentPropsWithoutRef<"td">) => (
      <td {...props} className={cn("px-2 py-1 align-top", props.className)} />
    ),
    a: (props: React.ComponentPropsWithoutRef<"a">) => (
      <a {...props} className={cn("underline underline-offset-2", props.className)} />
    ),
  };

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
            ? "max-w-[60%] rounded-2xl bg-surface px-4 py-2.5"
            : "w-full space-y-3",
        )}
      >
        <div
          className={cn(
            isUser ? "whitespace-pre-wrap" : "[&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
            isUser
              ? "text-[15px] leading-6 text-foreground"
              : "text-[16px] leading-[1.65] text-foreground [&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:py-0.5 [&_pre]:overflow-x-auto [&_pre]:rounded-xl [&_pre]:bg-muted [&_pre]:p-4",
          )}
        >
          {isUser ? (
            content
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {content}
            </ReactMarkdown>
          )}
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
