const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8000";
const sessionStorageKey = "gitawise_session_id";

function getOrCreateSessionId(): string {
  try {
    if (typeof localStorage !== "undefined") {
      const storedSessionId = localStorage.getItem(sessionStorageKey);
      if (storedSessionId) return storedSessionId;

      let sid: string;
      // Prefer crypto.randomUUID when available
      if (typeof (globalThis as any).crypto?.randomUUID === "function") {
        sid = (globalThis as any).crypto.randomUUID();
      } else {
        // Fallback simple UUIDv4
        sid = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
          const r = (Math.random() * 16) | 0,
            v = c === "x" ? r : (r & 0x3) | 0x8;
          return v.toString(16);
        });
      }
      localStorage.setItem(sessionStorageKey, sid);
      return sid;
    }
  } catch (e) {
    // ignore storage errors and generate ephemeral id
    if (typeof (globalThis as any).crypto?.randomUUID === "function") {
      return (globalThis as any).crypto.randomUUID();
    }
    return "session-ephemeral-" + Date.now().toString(36);
  }
  // server-side or unknown environment
  return "session-ephemeral-" + Date.now().toString(36);
}

export function getStoredSessionId(): string | null {
  try {
    if (typeof localStorage === "undefined") return null;
    return localStorage.getItem(sessionStorageKey);
  } catch {
    return null;
  }
}

export function resetStoredSessionId(): string | null {
  const sessionId = getStoredSessionId();
  try {
    if (typeof localStorage !== "undefined") {
      localStorage.removeItem(sessionStorageKey);
    }
  } catch {
    // Ignore storage failures; the next request can still use an ephemeral id.
  }
  return sessionId;
}

export async function clearChatSession(sessionId: string): Promise<void> {
  await fetch(`${backendUrl}/query-engine/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
    cache: "no-store",
    keepalive: true,
  });
}

export type ApiCitation = {
  chapter: number;
  verse: number;
  shloka: string;
  translation: string;
  interpretation: string;
  meaning?: string;
  summary?: string;
  topics: string[];
};

export type QueryAnswerResponse = {
  original_query: string;
  route: string;
  answer: string;
  cited_verses: string[];
  warnings: string[];
  contexts: ApiCitation[];
  used_rag: boolean;
};

export async function fetchQueryAnswer(
  query: string,
  signal?: AbortSignal,
): Promise<QueryAnswerResponse> {
  const session_id = getOrCreateSessionId();
  const response = await fetch(`${backendUrl}/query-engine/answer`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query, session_id }),
    cache: "no-store",
    signal,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Failed to fetch answer from backend.");
  }

  return (await response.json()) as QueryAnswerResponse;
}
