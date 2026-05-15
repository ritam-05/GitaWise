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
    <section
      className={cn(
        "flex w-full",
        isUser ? "justify-end" : "justify-start",
      )}
    >
      <div
        className={cn(
          "max-w-[760px]",
          isUser
            ? "rounded-2xl border border-border bg-panel px-5 py-4"
            : "space-y-5",
        )}
      >
        {!isUser && (
          <p className="text-xs font-medium uppercase tracking-[0.12em] text-muted">
            GitaWise
          </p>
        )}
        <p
          className={cn(
            "whitespace-pre-wrap text-[15px] leading-7 tracking-normal",
            isUser ? "text-foreground" : "text-foreground",
          )}
        >
          {content}
        </p>
        {citations.length > 0 && (
          <div className="grid gap-3 pt-1">
            {citations.map((citation) => (
              <CitationCard
                key={`${citation.chapter}-${citation.verse}`}
                {...citation}
              />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
