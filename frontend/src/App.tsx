import { useState, useEffect, useRef, useCallback } from 'react';
import { Message, ChatResponse, UserContext, TestCase } from './types';
import { sendMessage, checkHealth } from './api';
import { testCases, testCaseCategories } from './testCases';

type DebugTab = 'response' | 'request' | 'data';

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [selectedMessage, setSelectedMessage] = useState<Message | null>(null);
  const [debugTab, setDebugTab] = useState<DebugTab>('response');
  const [healthStatus, setHealthStatus] = useState<'checking' | 'ok' | 'error'>('checking');
  const [context, setContext] = useState<UserContext>({
    user_id: '',
    ui_state: {},
    source: 'test-frontend',
  });

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Check health on mount
  useEffect(() => {
    checkHealth()
      .then(() => setHealthStatus('ok'))
      .catch(() => setHealthStatus('error'));

    const interval = setInterval(() => {
      checkHealth()
        .then(() => setHealthStatus('ok'))
        .catch(() => setHealthStatus('error'));
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-select last message for debug
  useEffect(() => {
    if (messages.length > 0 && !selectedMessage) {
      const lastAssistant = [...messages].reverse().find(m => m.role === 'assistant');
      if (lastAssistant) {
        setSelectedMessage(lastAssistant);
      }
    }
  }, [messages, selectedMessage]);

  const handleSend = useCallback(async (text: string, testCase?: TestCase) => {
    if (!text.trim() || isLoading) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    const startTime = performance.now();

    const request = {
      message: text,
      conversation_id: conversationId || undefined,
      user_id: testCase?.user_id || context.user_id || undefined,
      source: context.source,
      ui_state: testCase?.ui_state || (Object.keys(context.ui_state).length > 0 ? context.ui_state : undefined),
    };

    try {
      const response: ChatResponse = await sendMessage(request);
      const latency = Math.round(performance.now() - startTime);

      if (!conversationId) {
        setConversationId(response.conversation_id);
      }

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: response.reply.text,
        timestamp: new Date(),
        request,
        response,
        latency,
      };

      setMessages(prev => [...prev, assistantMessage]);
      setSelectedMessage(assistantMessage);
    } catch (error) {
      const errorMessage: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        request,
        error: error instanceof Error ? error.message : 'Unknown error',
        latency: Math.round(performance.now() - startTime),
      };

      setMessages(prev => [...prev, errorMessage]);
      setSelectedMessage(errorMessage);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }, [conversationId, context, isLoading]);

  const handleTestCase = (tc: TestCase) => {
    handleSend(tc.message, tc);
  };

  const handleQuickReply = (label: string) => {
    handleSend(label);
  };

  const handleClear = () => {
    setMessages([]);
    setConversationId(null);
    setSelectedMessage(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend(inputValue);
    }
  };

  const formatJson = (obj: unknown): JSX.Element => {
    const json = JSON.stringify(obj, null, 2);
    return (
      <span dangerouslySetInnerHTML={{
        __html: json
          .replace(/"([^"]+)":/g, '<span class="json-key">"$1"</span>:')
          .replace(/: "([^"]*)"/g, ': <span class="json-string">"$1"</span>')
          .replace(/: (\d+\.?\d*)/g, ': <span class="json-number">$1</span>')
          .replace(/: (true|false)/g, ': <span class="json-boolean">$1</span>')
          .replace(/: (null)/g, ': <span class="json-null">$1</span>')
      }} />
    );
  };

  const groupedTestCases = testCaseCategories.map(category => ({
    category,
    cases: testCases.filter(tc => tc.category === category),
  }));

  return (
    <div className="app">
      {/* Sidebar with Test Cases */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>Pharmacy Assistant</h1>
          <p>Тестовый интерфейс v1.0</p>
        </div>
        
        <div className="test-cases">
          {groupedTestCases.map(group => (
            <div key={group.category} className="category-group">
              <div className="category-title">{group.category}</div>
              {group.cases.map(tc => (
                <button
                  key={tc.id}
                  className="test-case-btn"
                  onClick={() => handleTestCase(tc)}
                  disabled={isLoading}
                >
                  <span className="name">{tc.name}</span>
                  {tc.description && <span className="desc">{tc.description}</span>}
                </button>
              ))}
            </div>
          ))}
        </div>

        <div className="context-settings">
          <h3>Контекст запроса</h3>
          <div className="context-field">
            <label>User ID</label>
            <input
              type="text"
              value={context.user_id}
              onChange={e => setContext({ ...context, user_id: e.target.value })}
              placeholder="test-user-123"
            />
          </div>
          <div className="context-field">
            <label>Screen</label>
            <select
              value={context.ui_state.screen || ''}
              onChange={e => setContext({
                ...context,
                ui_state: { ...context.ui_state, screen: e.target.value || undefined }
              })}
            >
              <option value="">— не указан —</option>
              <option value="home">Home</option>
              <option value="catalog">Catalog</option>
              <option value="product">Product</option>
              <option value="cart">Cart</option>
              <option value="profile">Profile</option>
              <option value="orders">Orders</option>
            </select>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="chat-area">
        <header className="chat-header">
          <div className="chat-header-left">
            <div className="health-indicator">
              <span className={`health-dot ${healthStatus}`}></span>
              <span>
                {healthStatus === 'checking' && 'Проверка...'}
                {healthStatus === 'ok' && 'Backend подключён'}
                {healthStatus === 'error' && 'Backend недоступен'}
              </span>
            </div>
            {conversationId && (
              <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                ID: {conversationId.slice(0, 8)}...
              </span>
            )}
          </div>
          <button className="clear-btn" onClick={handleClear}>
            Очистить чат
          </button>
        </header>

        <div className="messages">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">💬</div>
              <h3>Начните диалог</h3>
              <p>Введите сообщение или выберите тест-кейс из списка слева для быстрого тестирования</p>
            </div>
          ) : (
            messages.map(msg => (
              <div
                key={msg.id}
                className={`message ${msg.role}`}
                onClick={() => msg.role === 'assistant' && setSelectedMessage(msg)}
                style={{ cursor: msg.role === 'assistant' ? 'pointer' : 'default' }}
              >
                {msg.error ? (
                  <div className="error-message">
                    ❌ Ошибка: {msg.error}
                  </div>
                ) : (
                  <div className="message-bubble">
                    <p>{msg.content}</p>
                  </div>
                )}
                
                {msg.role === 'assistant' && msg.response?.meta?.quick_replies && (
                  <div className="quick-replies">
                    {msg.response.meta.quick_replies.map((qr, idx) => (
                      <button
                        key={idx}
                        className="quick-reply-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleQuickReply(qr.label || qr.value);
                        }}
                      >
                        {qr.label || qr.value}
                      </button>
                    ))}
                  </div>
                )}

                <div className="message-meta">
                  <span>{msg.timestamp.toLocaleTimeString()}</span>
                  {msg.latency !== undefined && (
                    <span className="latency-badge">{msg.latency}ms</span>
                  )}
                  {msg.role === 'assistant' && selectedMessage?.id === msg.id && (
                    <span style={{ color: 'var(--accent-cyan)' }}>● selected</span>
                  )}
                </div>
              </div>
            ))
          )}
          
          {isLoading && (
            <div className="message assistant">
              <div className="message-bubble">
                <div className="typing-indicator">
                  <span className="typing-dot"></span>
                  <span className="typing-dot"></span>
                  <span className="typing-dot"></span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        <div className="input-area">
          <div className="input-wrapper">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Введите сообщение..."
              disabled={isLoading}
            />
            <button
              className="send-btn"
              onClick={() => handleSend(inputValue)}
              disabled={isLoading || !inputValue.trim()}
            >
              Отправить
            </button>
          </div>
        </div>
      </main>

      {/* Debug Panel */}
      <aside className="debug-panel">
        <header className="debug-header">
          <h2>Debug Info</h2>
          <div className="debug-tabs">
            <button
              className={`debug-tab ${debugTab === 'response' ? 'active' : ''}`}
              onClick={() => setDebugTab('response')}
            >
              Response
            </button>
            <button
              className={`debug-tab ${debugTab === 'request' ? 'active' : ''}`}
              onClick={() => setDebugTab('request')}
            >
              Request
            </button>
            <button
              className={`debug-tab ${debugTab === 'data' ? 'active' : ''}`}
              onClick={() => setDebugTab('data')}
            >
              Data
            </button>
          </div>
        </header>

        <div className="debug-content">
          {!selectedMessage ? (
            <div className="debug-empty">
              Выберите сообщение ассистента для просмотра деталей
            </div>
          ) : selectedMessage.error ? (
            <div className="debug-section">
              <div className="debug-section-title">❌ Error</div>
              <div className="json-viewer">{selectedMessage.error}</div>
            </div>
          ) : debugTab === 'response' ? (
            <>
              {selectedMessage.response?.meta?.confidence !== undefined && (
                <div className="debug-section">
                  <div className="debug-section-title">📊 Confidence</div>
                  <div className="confidence-meter">
                    <div className="confidence-bar">
                      <div
                        className={`confidence-fill ${
                          selectedMessage.response.meta.confidence >= 0.8 ? 'high' :
                          selectedMessage.response.meta.confidence >= 0.5 ? 'medium' : 'low'
                        }`}
                        style={{ width: `${selectedMessage.response.meta.confidence * 100}%` }}
                      />
                    </div>
                    <span className="confidence-value">
                      {(selectedMessage.response.meta.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              )}

              {selectedMessage.response?.meta?.top_intent && (
                <div className="debug-section">
                  <div className="debug-section-title">🎯 Top Intent</div>
                  <div className="json-viewer">{selectedMessage.response.meta.top_intent}</div>
                </div>
              )}

              {selectedMessage.response?.actions && selectedMessage.response.actions.length > 0 && (
                <div className="debug-section">
                  <div className="debug-section-title">⚡ Actions ({selectedMessage.response.actions.length})</div>
                  <div style={{ marginBottom: '8px' }}>
                    {selectedMessage.response.actions.map((action, idx) => (
                      <span key={idx} className={`action-badge ${action.channel || 'other'}`}>
                        {action.intent || action.type}
                      </span>
                    ))}
                  </div>
                  <div className="json-viewer">
                    {formatJson(selectedMessage.response.actions)}
                  </div>
                </div>
              )}

              {selectedMessage.response?.meta && (
                <div className="debug-section">
                  <div className="debug-section-title">📋 Meta</div>
                  <div className="json-viewer">
                    {formatJson(selectedMessage.response.meta)}
                  </div>
                </div>
              )}

              {selectedMessage.response?.reply && (
                <div className="debug-section">
                  <div className="debug-section-title">💬 Reply Object</div>
                  <div className="json-viewer">
                    {formatJson(selectedMessage.response.reply)}
                  </div>
                </div>
              )}
            </>
          ) : debugTab === 'request' ? (
            <div className="debug-section">
              <div className="debug-section-title">📤 Request</div>
              <div className="json-viewer">
                {formatJson(selectedMessage.request)}
              </div>
            </div>
          ) : (
            <>
              {selectedMessage.response?.data && (
                <>
                  {selectedMessage.response.data.products.length > 0 && (
                    <div className="debug-section">
                      <div className="debug-section-title">📦 Products ({selectedMessage.response.data.products.length})</div>
                      <div className="json-viewer">
                        {formatJson(selectedMessage.response.data.products)}
                      </div>
                    </div>
                  )}

                  {selectedMessage.response.data.orders.length > 0 && (
                    <div className="debug-section">
                      <div className="debug-section-title">🛒 Orders ({selectedMessage.response.data.orders.length})</div>
                      <div className="json-viewer">
                        {formatJson(selectedMessage.response.data.orders)}
                      </div>
                    </div>
                  )}

                  {selectedMessage.response.data.pharmacies.length > 0 && (
                    <div className="debug-section">
                      <div className="debug-section-title">🏪 Pharmacies ({selectedMessage.response.data.pharmacies.length})</div>
                      <div className="json-viewer">
                        {formatJson(selectedMessage.response.data.pharmacies)}
                      </div>
                    </div>
                  )}

                  {selectedMessage.response.data.cart && (
                    <div className="debug-section">
                      <div className="debug-section-title">🛍️ Cart</div>
                      <div className="json-viewer">
                        {formatJson(selectedMessage.response.data.cart)}
                      </div>
                    </div>
                  )}

                  {selectedMessage.response.data.favorites.length > 0 && (
                    <div className="debug-section">
                      <div className="debug-section-title">❤️ Favorites ({selectedMessage.response.data.favorites.length})</div>
                      <div className="json-viewer">
                        {formatJson(selectedMessage.response.data.favorites)}
                      </div>
                    </div>
                  )}

                  {Object.keys(selectedMessage.response.data.metadata).length > 0 && (
                    <div className="debug-section">
                      <div className="debug-section-title">📎 Metadata</div>
                      <div className="json-viewer">
                        {formatJson(selectedMessage.response.data.metadata)}
                      </div>
                    </div>
                  )}

                  {!selectedMessage.response.data.products.length &&
                   !selectedMessage.response.data.orders.length &&
                   !selectedMessage.response.data.pharmacies.length &&
                   !selectedMessage.response.data.cart &&
                   !selectedMessage.response.data.favorites.length &&
                   Object.keys(selectedMessage.response.data.metadata).length === 0 && (
                    <div className="debug-empty">
                      Нет данных в ответе
                    </div>
                  )}
                </>
              )}
            </>
          )}
        </div>
      </aside>
    </div>
  );
}

export default App;

