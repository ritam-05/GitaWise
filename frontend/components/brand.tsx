import Link from "next/link";

export function Brand() {
  return (
    <Link href="/" className="inline-flex items-center gap-3">
      <span className="flex h-8 w-8 items-center justify-center rounded-full border border-border bg-surface text-[16.5px] font-medium text-foreground">
        G
      </span>
      <span className="text-[16.5px] font-medium tracking-normal text-foreground">
        GitaWise
      </span>
    </Link>
  );
}
