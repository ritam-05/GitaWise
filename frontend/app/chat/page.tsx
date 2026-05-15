import { Suspense } from "react";
import { ChatInput } from "@/components/chat-input";
import { ChatMessage, type Citation } from "@/components/chat-message";
import { Sidebar } from "@/components/sidebar";
import { Brand } from "@/components/brand";

const sampleCitations: Citation[] = [
  {
    chapter: 2,
    verse: 47,
    shloka: "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ।",
    meaning:
      "You have a right to perform your prescribed duties, but you are not entitled to the fruits of your actions.",
    tags: ["action", "detachment", "discipline"],
  },
  {
    chapter: 2,
    verse: 48,
    shloka: "योगस्थः कुरु कर्माणि सङ्गं त्यक्त्वा धनञ्जय ।",
    meaning:
      "Established in yoga, perform your actions, abandoning attachment and remaining even-minded in success and failure.",
    tags: ["equanimity", "yoga", "steadiness"],
  },
];

function ChatContent({
  question,
}: {
  question: string;
}) {
  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />
      <main className="flex min-h-screen flex-1 flex-col">
        <header className="sticky top-0 z-10 border-b border-border bg-background px-5 py-4">
          <div className="mx-auto flex max-w-4xl items-center justify-between">
            <div className="lg:hidden">
              <Brand />
            </div>
            <p className="hidden text-sm text-muted lg:block">
              GitaWise answers should remain grounded in cited verses.
            </p>
            <span className="rounded-full border border-border px-3 py-1 text-xs text-muted">
              BGE-M3 ready
            </span>
          </div>
        </header>

        <div className="subtle-scrollbar flex-1 overflow-y-auto px-5 py-8">
          <div className="mx-auto w-full max-w-4xl space-y-10">
            <ChatMessage role="user" content={question} />
            <ChatMessage
              role="assistant"
              content={
                "The Gita frames detached action as a discipline of attention, not indifference. You still act carefully, ethically, and with full effort, but you stop making inner stability depend on the outcome. The work remains yours; the result is shaped by conditions beyond the self."
              }
              citations={sampleCitations}
            />
          </div>
        </div>

        <div className="border-t border-border bg-background px-5 py-4">
          <div className="mx-auto max-w-4xl">
            <ChatInput compact initialValue="" />
          </div>
        </div>
      </main>
    </div>
  );
}

export default async function ChatPage({
  searchParams,
}: {
  searchParams?: Promise<{ q?: string }>;
}) {
  const params = await searchParams;
  const question =
    params?.q?.trim() ||
    "How can I act sincerely without becoming attached to the result?";

  return (
    <Suspense>
      <ChatContent question={question} />
    </Suspense>
  );
}
