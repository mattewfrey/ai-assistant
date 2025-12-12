// API Request/Response types matching backend models

export interface UIState {
  entry_point?: string;
  screen?: string;
  selected_region_id?: string;
  selected_pharmacy_id?: string;
}

export interface ChatMeta {
  client_version?: string;
  device?: string;
}

export interface ChatRequest {
  conversation_id?: string;
  trace_id?: string;
  message: string;
  user_id?: string;
  source?: string;
  ui_state?: UIState;
  meta?: ChatMeta;
}

export interface Reply {
  text: string;
  tone?: string;
  title?: string;
  display_hints?: Record<string, unknown>;
}

export interface AssistantAction {
  type: string;
  channel?: string;
  intent?: string;
  parameters: Record<string, unknown>;
}

export interface AssistantMeta {
  confidence?: number;
  legal_disclaimer?: string;
  quick_replies?: Array<{ label: string; value: string }>;
  top_intent?: string;
  extracted_entities?: Record<string, unknown>;
  debug?: Record<string, unknown>;
}

export interface DataPayload {
  products: Array<Record<string, unknown>>;
  cart?: Record<string, unknown>;
  orders: Array<Record<string, unknown>>;
  user_profile?: Record<string, unknown>;
  user_addresses: Array<Record<string, unknown>>;
  favorites: Array<Record<string, unknown>>;
  pharmacies: Array<Record<string, unknown>>;
  recommendations: Array<Record<string, unknown>>;
  message?: string;
  metadata: Record<string, unknown>;
}

export interface ChatResponse {
  conversation_id: string;
  reply: Reply;
  actions: AssistantAction[];
  meta?: AssistantMeta;
  data: DataPayload;
  ui_state?: Record<string, unknown>;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  request?: ChatRequest;
  response?: ChatResponse;
  error?: string;
  latency?: number;
}

export interface TestCase {
  id: string;
  name: string;
  category: string;
  message: string;
  description?: string;
  user_id?: string;
  ui_state?: UIState;
}

export interface UserContext {
  user_id: string;
  ui_state: UIState;
  source: string;
}

