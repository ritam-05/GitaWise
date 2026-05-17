"use client";

import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";

type DisclaimerModalProps = {
  isOpen: boolean;
  onClose: () => void;
};

export function DisclaimerModal({ isOpen, onClose }: DisclaimerModalProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.6 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
            className="fixed inset-0 z-40 bg-black"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
          >
            <div className="relative max-h-[80vh] w-full max-w-2xl overflow-y-auto rounded-2xl bg-card p-6 shadow-[0_20px_60px_rgba(0,0,0,0.3)]">
              {/* Close Button */}
              <button
                type="button"
                onClick={onClose}
                className="absolute left-4 top-4 inline-flex h-8 w-8 items-center justify-center rounded-full border border-transparent text-muted transition-colors hover:border-border hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
                aria-label="Close disclaimer"
              >
                <X className="h-5 w-5" aria-hidden="true" />
              </button>

              {/* Title */}
              <div className="pr-12">
                <h2 className="text-center text-2xl font-semibold text-foreground mb-4">
                  Disclaimer
                </h2>

                {/* Content */}
                <div className="space-y-4 text-[15px] leading-7 text-secondary">
                  <p>
                    GitaWise is an AI-powered educational and spiritual companion inspired by the Bhagavad Gita, and does not, anywhere, claim to be LORD KRISHNA himself. Responses are generated using AI and scripture retrieval systems and are intended for informational, reflective, and personal growth purposes only.
                  </p>

                  <p>
                    The platform does not provide medical, psychological, legal, or professional advice, and it does not claim to represent or speak as Lord Krishna or any religious authority. All responses include chapter and verse references, but interpretations may vary from traditional teachings.
                  </p>

                  <p>
                    Users are encouraged to consult authentic translations, scholars, or professionals where appropriate. By using GitaWise, you acknowledge that AI-generated responses may occasionally contain inaccuracies or limitations.
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
