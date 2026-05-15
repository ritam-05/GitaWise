import { ChatInput } from "@/components/chat-input";
import { FadeIn } from "@/components/fade-in";
import { PromptSuggestions } from "@/components/prompt-suggestions";
import { TopNav } from "@/components/top-nav";

const notes = [
  "Verse-grounded answers",
  "Philosophical interpretation",
  "Calm, citation-first retrieval",
];

export default function HomePage() {
  return (
    <main className="min-h-screen bg-background">
      <TopNav />
      <section className="mx-auto flex min-h-[calc(100vh-84px)] w-full max-w-4xl flex-col items-center justify-center px-5 pb-16 pt-8">
        <FadeIn>
          <div className="mx-auto max-w-2xl space-y-4">
            <p className="text-sm font-medium uppercase tracking-[0.16em] text-muted">
              Bhagavad Gita assistant
            </p>
            <h1 className="font-goudy text-4xl font-medium tracking-normal text-foreground sm:text-5xl">
              Gita-Wise
            </h1>
            <p className="mx-auto max-w-xl text-base leading-7 text-secondary">
              Ask a question about action, duty, fear, attention, or peace.
              Receive a grounded answer with verses kept close to the surface.
            </p>
          </div>

          <div className="mx-auto w-full max-w-2xl">
            <ChatInput />
          </div>

          <PromptSuggestions />

          <div className="mx-auto grid max-w-2xl gap-3 border-t border-border pt-8 text-left sm:grid-cols-3">
            {notes.map((note) => (
              <div key={note} className="text-sm leading-6 text-muted">
                {note}
              </div>
            ))}
          </div>
        </FadeIn>
      </section>
    </main>
  );
}
