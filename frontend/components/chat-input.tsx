"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowUp, BookOpen, CornerDownLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

type ChatInputProps = {
  compact?: boolean;
  initialValue?: string;
  className?: string;
};

export function ChatInput({
  compact = false,
  initialValue = "",
  className,
}: ChatInputProps) {
  const [value, setValue] = useState(initialValue);
  const router = useRouter();

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const query = value.trim();

    if (!query) {
      return;
    }

    router.push(`/chat?q=${encodeURIComponent(query)}`);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className={cn(
        "rounded-3xl border border-border bg-surface p-2 shadow-none transition-colors",
        className,
      )}
    >
      <Textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        placeholder="Ask about duty, doubt, peace, grief, action..."
        className={cn(
          "border-0 bg-surface px-3 py-3 shadow-none focus:ring-0",
          compact ? "min-h-[60px] text-sm" : "min-h-[112px]",
        )}
        onKeyDown={(event) => {
          if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
            event.currentTarget.form?.requestSubmit();
          }
        }}
      />
      <div className="flex items-center justify-between gap-3 rounded-2xl bg-surface px-2 pb-1 transition-colors">
        <div className="flex items-center gap-2 text-xs text-muted">
          <BookOpen className="h-3.5 w-3.5" aria-hidden="true" />
          <span>Grounded in verse citations</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="hidden items-center gap-1 text-xs text-muted sm:inline-flex">
            <CornerDownLeft className="h-3 w-3" aria-hidden="true" />
            Ctrl Enter
          </span>
          <Button
            type="submit"
            size="icon"
            variant="primary"
            aria-label="Send message"
            disabled={!value.trim()}
          >
            <ArrowUp className="h-4 w-4" aria-hidden="true" />
          </Button>
        </div>
      </div>
    </form>
  );
}
