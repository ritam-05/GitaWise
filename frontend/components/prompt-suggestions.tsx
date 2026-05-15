import Link from "next/link";

const prompts = [
  "How do I act without attachment?",
  "What does the Gita say about fear?",
  "Explain dharma in a practical way",
  "How can I steady my mind?",
];

export function PromptSuggestions() {
  return (
    <div className="flex flex-wrap items-center justify-center gap-2">
      {prompts.map((prompt) => (
        <Link
          key={prompt}
          href={`/chat?q=${encodeURIComponent(prompt)}`}
          className="rounded-full border border-border bg-surface px-3 py-2 text-sm text-secondary transition-colors hover:bg-card hover:text-foreground"
        >
          {prompt}
        </Link>
      ))}
    </div>
  );
}
