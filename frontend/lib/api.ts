const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? "Something went wrong. Please try again.");
  }
  return response.status === 204 ? (undefined as T) : response.json() as Promise<T>;
}

export type DocumentStatus = "processing" | "ready" | "empty" | "failed";
export type DocumentItem = { id: string; filename: string; page_count: number; status: DocumentStatus; status_detail: string | null; created_at: string };
export type Citation = { index: number; document_id: string; filename: string; page_number: number; excerpt: string };
export type ChatResult = { conversation_id: string; answer: string; citations: Citation[] };

export const listDocuments = () => request<DocumentItem[]>("/api/v1/documents");
export const getSystemStatus = () => request<{ ollama: { available: boolean; generation_model_ready: boolean; embedding_model_ready: boolean } }>("/api/v1/system/status");
export async function uploadDocument(file: File) {
  const form = new FormData();
  form.append("file", file);
  return request<DocumentItem>("/api/v1/documents", { method: "POST", body: form });
}
export const deleteDocument = (id: string) => request<void>(`/api/v1/documents/${id}`, { method: "DELETE" });
export const askQuestion = (question: string, document_ids: string[], conversation_id?: string) => request<ChatResult>("/api/v1/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, document_ids, conversation_id }) });
