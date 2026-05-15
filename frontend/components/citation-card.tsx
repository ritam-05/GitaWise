type CitationCardProps = {
  chapter: number;
  verse: number;
  shloka: string;
  meaning: string;
  tags: string[];
};

export function CitationCard({
  chapter,
  verse,
  shloka,
  meaning,
  tags,
}: CitationCardProps) {
  return (
    <article className="rounded-2xl border border-border bg-surface p-5">
      <div className="mb-4 flex items-center justify-between gap-3">
        <p className="text-xs font-medium uppercase tracking-[0.12em] text-muted">
          Chapter {chapter} · Verse {verse}
        </p>
        <span className="rounded-full border border-border px-2.5 py-1 text-xs text-muted">
          Citation
        </span>
      </div>
      <blockquote className="text-base leading-relaxed text-foreground">
        {shloka}
      </blockquote>
      <p className="mt-4 text-sm leading-relaxed text-secondary">{meaning}</p>
      <div className="mt-5 flex flex-wrap gap-2">
        {tags.map((tag) => (
          <span
            key={tag}
            className="rounded-full bg-panel px-2.5 py-1 text-xs text-muted"
          >
            {tag}
          </span>
        ))}
      </div>
    </article>
  );
}
