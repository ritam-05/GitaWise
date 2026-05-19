"use client";

import { useState } from "react";
import { Rnd } from "react-rnd";
import { GripHorizontal } from "lucide-react";
import { ChatInput } from "@/components/chat-input";

type FloatingInputProps = {
  onSubmit?: (query: string) => void;
  onStop?: () => void;
  isGenerating?: boolean;
  disabled?: boolean;
};

export function FloatingInput({
  onSubmit,
  onStop,
  isGenerating = false,
  disabled = false,
}: FloatingInputProps) {
  const [showContent, setShowContent] = useState(true);

  return (
    <Rnd
      default={{
        x: window.innerWidth - 520,
        y: window.innerHeight - 120,
        width: 500,
        height: "auto",
      }}
      minWidth={80}
      maxWidth={800}
      minHeight={80}
      maxHeight={window.innerHeight - 40}
      dragHandleClassName="drag-handle-input"
      bounds="window"
      enableResizing={{
        top: true,
        right: true,
        bottom: true,
        left: true,
        topRight: true,
        bottomRight: true,
        bottomLeft: true,
        topLeft: true,
      }}
      onResizeStop={(e, direction, ref) => {
        const width = ref.offsetWidth;
        const height = ref.offsetHeight;
        setShowContent(width > 150 || height > 150);
      }}
      style={{
        position: "fixed",
        zIndex: 40,
      }}
    >
      <div className="rounded-2xl border border-white/10 bg-surface shadow-lg h-full flex flex-col">
        {/* Drag handle header */}
        <div className="drag-handle-input flex items-center gap-2 border-b border-white/10 bg-surface px-3 py-2 cursor-move shrink-0">
          <GripHorizontal className="h-4 w-4 text-muted opacity-40" />
        </div>

        {/* Input - hidden when too small */}
        {showContent && (
          <div className="p-3 overflow-hidden">
            <ChatInput
              compact
              onSubmit={onSubmit}
              onStop={onStop}
              isGenerating={isGenerating}
              disabled={disabled}
            />
          </div>
        )}
      </div>
    </Rnd>
  );
}
