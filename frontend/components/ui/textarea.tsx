import * as React from "react";
import { cn } from "@/lib/utils";

export interface TextareaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => (
    <textarea
      ref={ref}
      className={cn(
        "min-h-[96px] w-full rounded-2xl border border-border bg-surface px-4 py-4 text-[18px] leading-relaxed text-foreground placeholder:text-muted shadow-none transition-colors focus:outline-none focus:ring-1 focus:ring-accent",
        className,
      )}
      {...props}
    />
  ),
);

Textarea.displayName = "Textarea";

export { Textarea };
