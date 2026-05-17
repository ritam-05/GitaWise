"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowUp, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

type ChatInputProps = {
  compact?: boolean;
  initialValue?: string;
  className?: string;
  onSubmit?: (query: string) => void;
  onStop?: () => void;
  isGenerating?: boolean;
  disabled?: boolean;
};

export function ChatInput({
  compact = false,
  initialValue = "",
  className,
  onSubmit: onSubmitProp,
  onStop,
  isGenerating = false,
  disabled = false,
}: ChatInputProps) {
  const [value, setValue] = useState(initialValue);
  const router = useRouter();

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const query = value.trim();

    if (!query) {
      return;
    }

    // If onSubmit callback is provided, use it (for conversation state)
    if (onSubmitProp) {
      onSubmitProp(query);
      setValue("");
    } else {
      // Otherwise fall back to URL navigation
      router.push(`/chat?q=${encodeURIComponent(query)}`);
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className={cn(
        "flex items-end gap-2 rounded-2xl bg-surface p-2.5 shadow-none transition-colors",
        disabled && "opacity-60",
        className,
      )}
    >
      <button
        type="button"
        disabled={disabled}
        className="flex h-8 w-8 items-center justify-center rounded-full text-muted hover:bg-surface/60 hover:text-secondary transition disabled:opacity-50"
        aria-label="Attachment"
      >
        <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
        </svg>
      </button>
      <Textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        placeholder="Ask about dharma, action, wisdom…"
        disabled={disabled}
        className="flex-1 border-0 bg-transparent px-0 py-2 text-[15px] shadow-none focus:ring-0 placeholder:text-muted placeholder:opacity-50 disabled:opacity-50"
        style={{ minHeight: "36px", maxHeight: "120px" }}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            // Shift+Enter should insert a newline
            if (event.shiftKey) {
              return;
            }

            // Prevent submission when disabled or generating
            if (disabled || isGenerating) {
              event.preventDefault();
              return;
            }

            // Submit on Enter (also supports Ctrl/Cmd+Enter)
            event.preventDefault();
            event.currentTarget.form?.requestSubmit();
          }
        }}
      />
      {isGenerating ? (
        <Button
          type="button"
          size="icon"
          variant="primary"
          aria-label="Stop generation"
          onClick={onStop}
          className="h-8 w-8 rounded-full flex-shrink-0"
        >
          <Square className="h-3.5 w-3.5 fill-current" aria-hidden="true" />
        </Button>
      ) : (
        <Button
          type="submit"
          size="icon"
          variant="primary"
          aria-label="Send message"
          disabled={!value.trim() || disabled}
          className="h-8 w-8 rounded-full flex-shrink-0"
        >
          <ArrowUp className="h-3.5 w-3.5" aria-hidden="true" />
        </Button>
      )}
    </form>
  );
}
