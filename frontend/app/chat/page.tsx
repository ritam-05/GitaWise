"use client";

import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";

import { ChatInput } from "@/components/chat-input";
import { ChatMessage, type Citation } from "@/components/chat-message";
import { Sidebar } from "@/components/sidebar";
import { fetchQueryAnswer } from "@/lib/api";

type Message = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
};

export default function ChatPage() {
  const searchParams = useSearchParams();
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const hasInitialized = useRef(false);

  // Initialize with URL search param on mount (only once)
  useEffect(() => {
    if (hasInitialized.current) return;

    const question = searchParams?.get("q")?.trim();
    if (question) {
      hasInitialized.current = true;
      handleSubmit(question);
    }
  }, []);

  const handleSubmit = async (query: string) => {
    // Add user message
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setIsLoading(true);

    try {
      const result = await fetchQueryAnswer(query);
      const citations = result.contexts.map((context) => ({
        chapter: context.chapter,
        verse: context.verse,
        shloka: context.shloka,
        meaning: context.translation || context.interpretation,
        tags: context.topics,
      }));

      // Add assistant message
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: result.answer,
          citations: citations,
        },
      ]);
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? `Backend connection issue: ${error.message}`
          : "Backend connection issue.";

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: errorMessage,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />
      <main className="flex min-h-screen flex-1 flex-col">
        <div className="subtle-scrollbar flex-1 overflow-y-auto px-0 py-10">
          <div className="mx-auto w-full max-w-[60vw] space-y-6">
            {messages.length === 0 ? (
              <div className="flex h-full items-center justify-center">
                <p className="text-center text-muted">
                  Ask me anything about the Bhagavad Gita...
                </p>
              </div>
            ) : (
              messages.map((message, index) => (
                <ChatMessage
                  key={index}
                  role={message.role}
                  content={message.content}
                  citations={message.citations}
                />
              ))
            )}
            {isLoading && (
              <ChatMessage
                role="assistant"
                content="Thinking..."
                citations={[]}
              />
            )}
          </div>
        </div>

        <div className="sticky bottom-0 bg-background/86 px-0 py-5 backdrop-blur-xl">
          <div className="mx-auto w-full max-w-[60vw]">
            <ChatInput compact onSubmit={handleSubmit} disabled={isLoading} />
          </div>
        </div>
      </main>
    </div>
  );
}
