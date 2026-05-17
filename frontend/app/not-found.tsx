import Link from "next/link";

export default function NotFound() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-background px-6 text-center">
      <div className="max-w-sm space-y-4">
        <p className="text-[15px] text-muted">404</p>
        <h1 className="text-2xl font-medium tracking-normal text-foreground">
          This reflection is not here.
        </h1>
        <Link
          href="/"
          className="inline-flex rounded-full border border-border px-4 py-2 text-[15px] text-secondary transition-colors hover:bg-surface hover:text-foreground"
        >
          Return home
        </Link>
      </div>
    </main>
  );
}
