import Link from "next/link";
import { MessageSquare, Search, Settings } from "lucide-react";
import { Brand } from "@/components/brand";

const items = [
  { label: "Ask", href: "/chat", icon: MessageSquare },
  { label: "Explore", href: "/chat?q=Explain dharma", icon: Search },
  { label: "Preferences", href: "/chat", icon: Settings },
];

export function Sidebar() {
  return (
    <aside className="hidden min-h-screen w-64 shrink-0 border-r border-border bg-background px-4 py-5 lg:block">
      <Brand />
      <nav className="mt-10 space-y-1">
        {items.map((item) => {
          const Icon = item.icon;

          return (
            <Link
              key={item.label}
              href={item.href}
              className="flex items-center gap-3 rounded-xl px-3 py-2 text-sm text-secondary transition hover:bg-surface hover:text-foreground"
            >
              <Icon className="h-4 w-4" aria-hidden="true" />
              {item.label}
            </Link>
          );
        })}
      </nav>
      <div className="mt-10 rounded-2xl border border-border bg-surface p-4">
        <p className="text-xs font-medium uppercase tracking-[0.12em] text-muted">
          Retrieval mode
        </p>
        <p className="mt-3 text-sm leading-6 text-secondary">
          Verse-grounded responses with citations prepared for reranking.
        </p>
      </div>
    </aside>
  );
}
