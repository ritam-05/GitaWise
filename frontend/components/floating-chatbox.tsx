"use client";

import { Rnd } from "react-rnd";
import { useState } from "react";
import { X, GripHorizontal } from "lucide-react";

type FloatingChatBoxProps = {
  children: React.ReactNode;
  onClose?: () => void;
};

export function FloatingChatBox({ children, onClose }: FloatingChatBoxProps) {
  const [isMinimized, setIsMinimized] = useState(false);

  return (
    <Rnd
      default={{
        x: window.innerWidth - 520,
        y: 80,
        width: 500,
        height: 600,
      }}
      minWidth={300}
      minHeight={300}
      maxWidth={800}
      maxHeight={window.innerHeight - 40}
      dragHandleClassName="drag-handle"
      bounds="window"
      style={{
        position: "fixed",
        zIndex: 50,
      }}
    >
      <div className="flex h-full flex-col rounded-2xl border border-white/10 bg-background shadow-2xl">
        {/* Header with drag handle */}
        <div className="drag-handle flex items-center justify-between border-b border-white/10 bg-surface px-4 py-3 cursor-move">
          <div className="flex items-center gap-2">
            <GripHorizontal className="h-4 w-4 text-muted opacity-50" />
            <h2 className="text-sm font-semibold text-foreground">GitaWise Chat</h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsMinimized(!isMinimized)}
              className="p-1 hover:bg-surface rounded transition-colors"
              title={isMinimized ? "Maximize" : "Minimize"}
            >
              <svg className="h-4 w-4 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {isMinimized ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v12m-6-6h12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                )}
              </svg>
            </button>
            {onClose && (
              <button
                onClick={onClose}
                className="p-1 hover:bg-surface rounded transition-colors"
                title="Close"
              >
                <X className="h-4 w-4 text-muted" />
              </button>
            )}
          </div>
        </div>

        {/* Content */}
        {!isMinimized && (
          <div className="flex-1 overflow-hidden">
            {children}
          </div>
        )}
      </div>
    </Rnd>
  );
}
