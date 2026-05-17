import { HeroLineArt } from "@/components/hero-line-art";
import { ChatInput } from "@/components/chat-input";
import { FadeIn } from "@/components/fade-in";

export default function HomePage() {
  return (
    <main className="min-h-screen bg-background">
      <section className="mx-auto flex min-h-[calc(100vh-72px)] w-full max-w-4xl flex-col items-center justify-start px-5 pb-12 pt-0 sm:pt-1 lg:pt-2">
        <FadeIn>
          <div className="mx-auto max-w-xl space-y-2.5 sm:space-y-3">
            <HeroLineArt />
            <h1 className="font-goudy text-3xl font-medium tracking-normal text-foreground sm:text-4xl">
              Gita-Wise
            </h1>
            <p className="mx-auto max-w-lg text-[15px] leading-6 text-secondary sm:text-[18px]\">
              Ask a question about action, duty, fear, attention, or peace. Find
              clarity through Bhagwad Gita.
            </p>
          </div>

          <div className="mx-auto w-full max-w-xl">
            <ChatInput />
          </div>
        </FadeIn>
      </section>
    </main>
  );
}
