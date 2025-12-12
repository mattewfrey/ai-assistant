import { ChatRequest, ChatResponse } from './types';

const API_BASE = '/api/ai/chat';

export async function sendMessage(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/message`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.reason || `HTTP ${response.status}: ${response.statusText}`);
  }

  return response.json();
}

export async function checkHealth(): Promise<{
  status: string;
  checks: Record<string, boolean>;
  version: string;
}> {
  const response = await fetch('/health');
  if (!response.ok) {
    throw new Error('Backend is not available');
  }
  return response.json();
}

