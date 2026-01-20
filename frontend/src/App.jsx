// App.jsx - Main React component for Educational RAG Chatbot
import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE_URL = 'http://10.197.227.204:5000';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chatbotStatus, setChatbotStatus] = useState('disconnected');
  const [dbInfo, setDbInfo] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  // Check health and initialize chatbot on component mount
  useEffect(() => {
    checkHealth();
    initializeChatbot();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/`);
      const data = await response.json();
      setChatbotStatus(data.chatbot_ready ? 'ready' : 'needs-init');
    } catch (error) {
      console.error('Health check failed:', error);
      setChatbotStatus('disconnected');
    }
  };

  const initializeChatbot = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/initialize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({})
      });

      const data = await response.json();
      if (data.success) {
        setChatbotStatus('ready');
        loadDatabaseInfo();
        addSystemMessage('Chatbot initialized successfully! Ready to answer your questions.');
      } else {
        setChatbotStatus('error');
        addSystemMessage('Failed to initialize chatbot. Please check the knowledge base.');
      }
    } catch (error) {
      console.error('Initialization failed:', error);
      setChatbotStatus('error');
      addSystemMessage('Connection error. Please make sure the backend server is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const loadDatabaseInfo = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/database/info`);
      const data = await response.json();
      if (data.success) {
        setDbInfo(data);
      }
    } catch (error) {
      console.error('Failed to load database info:', error);
    }
  };

  const addSystemMessage = (text) => {
    const systemMessage = {
      id: Date.now(),
      text,
      sender: 'system',
      timestamp: new Date().toLocaleTimeString()
    };
    setMessages(prev => [...prev, systemMessage]);
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();

    if (!inputMessage.trim() || isLoading || chatbotStatus !== 'ready') {
      return;
    }

    const userMessage = {
      id: Date.now(),
      text: inputMessage,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: inputMessage,
          num_sources: 3
        })
      });

      const data = await response.json();

      const botMessage = {
        id: Date.now() + 1,
        text: data.answer || 'Sorry, I could not generate a response.',
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
        confidence: data.confidence,
        sources: data.sources || []
      };

      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now() + 1,
        text: 'Sorry, there was an error processing your request. Please try again.',
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const getStatusColor = () => {
    switch (chatbotStatus) {
      case 'ready': return '#4CAF50';
      case 'needs-init': return '#FF9800';
      case 'error': return '#F44336';
      default: return '#9E9E9E';
    }
  };

  const getStatusText = () => {
    switch (chatbotStatus) {
      case 'ready': return 'Ready';
      case 'needs-init': return 'Initializing...';
      case 'error': return 'Error';
      default: return 'Disconnected';
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>RAG Chatbot</h1>
        <div className="status-indicator">
          <div 
            className="status-dot" 
            style={{ backgroundColor: getStatusColor() }}
          ></div>
          <span>{getStatusText()}</span>
        </div>
      </header>

      {dbInfo && (
        <div className="database-info">
          <p><strong>Knowledge Base:</strong> {dbInfo.total_chunks} chunks • {dbInfo.embedding_model}</p>
        </div>
      )}

      <div className="chat-container">
        <div className="messages">
          {messages.length === 0 && (
            <div className="welcome-message">
              <h3>Welcome to the Educational RAG Chatbot!</h3>
              <p>Ask me any questions about your educational materials.</p>
              <p>I'll search through the knowledge base to provide accurate answers.</p>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`message ${message.sender}`}>
              <div className="message-content">
                <p>{message.text}</p>
                {message.confidence && (
                  <div className="message-meta">
                    <span className={`confidence confidence-${message.confidence}`}>
                      {message.confidence.toUpperCase()} CONFIDENCE
                    </span>
                    
                  </div>
                )}
                <span className="timestamp">{message.timestamp}</span>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message bot loading">
              <div className="message-content">
                <div className="typing-indicator">
                  <div></div><div></div><div></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSendMessage} className="input-form">
          <div className="input-container">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder={
                chatbotStatus === 'ready' 
                  ? "Ask a question about your educational materials..." 
                  : "Please wait for chatbot to initialize..."
              }
              disabled={isLoading || chatbotStatus !== 'ready'}
              className="message-input"
            />
            <button 
              type="submit" 
              disabled={isLoading || chatbotStatus !== 'ready' || !inputMessage.trim()}
              className="send-button"
            >

              {isLoading ? '⏳' : '📤'}
            </button>
          </div>
        </form>

        <div className="chat-controls">
          <button onClick={clearChat} className="clear-button">
            🗑️ Clear Chat
          </button>
          <button onClick={initializeChatbot} className="reinit-button">
            🔄 Reinitialize
          </button>
        </div>
      </div>
    </div>
  );
}
export default App;
/*line206<span className="sources-count">
                      {message.sources?.length || 0} sources
                    </span>*/

/*// App.jsx - CORRECTED version for Quality-Optimized Backend
import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE_URL = 'http://localhost:5000';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chatbotStatus, setChatbotStatus] = useState('disconnected');
  const [dbInfo, setDbInfo] = useState(null);
  const [performanceStats, setPerformanceStats] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  // Check health and load info on component mount
  useEffect(() => {
    checkHealth();
    loadDatabaseInfo();
    loadPerformanceStats();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/`);
      const data = await response.json();
      
      console.log('Health check response:', data); // Debug log
      
      setChatbotStatus(data.chatbot_ready ? 'ready' : 'needs-init');
      
      if (data.chatbot_ready) {
        addSystemMessage('✅ Educational RAG Chatbot is ready! Ask me anything about your educational materials.');
      } else {
        addSystemMessage('⚠️ Chatbot is initializing. Please wait...');
      }
    } catch (error) {
      console.error('Health check failed:', error);
      setChatbotStatus('disconnected');
      addSystemMessage('❌ Cannot connect to backend server. Please make sure it\'s running on port 5000.');
    }
  };

  const loadDatabaseInfo = async () => {
    try {
      // CORRECTED: Use /api/database/info (not /database/info)
      const response = await fetch(`${API_BASE_URL}/api/database/info`);
      const data = await response.json();
      
      console.log('Database info response:', data); // Debug log
      
      if (data.success) {
        setDbInfo(data.database_info);
      }
    } catch (error) {
      console.error('Failed to load database info:', error);
    }
  };

  const loadPerformanceStats = async () => {
    try {
      // NEW: Load performance statistics
      const response = await fetch(`${API_BASE_URL}/api/performance`);
      const data = await response.json();
      
      console.log('Performance stats response:', data); // Debug log
      
      if (data.success) {
        setPerformanceStats(data.performance_stats);
      }
    } catch (error) {
      console.error('Failed to load performance stats:', error);
    }
  };

  const addSystemMessage = (text) => {
    const systemMessage = {
      id: Date.now(),
      text,
      sender: 'system',
      timestamp: new Date().toLocaleTimeString()
    };
    setMessages(prev => [...prev, systemMessage]);
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();

    if (!inputMessage.trim() || isLoading) {
      return;
    }

    // Check if chatbot is ready
    if (chatbotStatus !== 'ready') {
      addSystemMessage('⚠️ Please wait for the chatbot to be ready before asking questions.');
      return;
    }

    const userMessage = {
      id: Date.now(),
      text: inputMessage,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentQuestion = inputMessage;
    setInputMessage('');
    setIsLoading(true);

    try {
      console.log('Sending question:', currentQuestion); // Debug log

      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: currentQuestion,
          num_sources: 5  // CORRECTED: Use 5 sources for quality
        })
      });

      const data = await response.json();
      
      console.log('Chat response:', data); // Debug log - This will show you what the backend actually returns

      if (data.success) {
        const botMessage = {
          id: Date.now() + 1,
          text: data.answer || 'Sorry, I could not generate a response.',
          sender: 'bot',
          timestamp: new Date().toLocaleTimeString(),
          confidence: data.confidence,
          sources_count: data.sources_count || 0,  // CORRECTED: Use sources_count from backend
          sources: data.sources || [],  // This should now work if backend sends sources array
          response_time: data.response_time_ms,
          relevance_score: data.relevance_score
        };

        setMessages(prev => [...prev, botMessage]);
        
        // Update performance stats after each query
        loadPerformanceStats();
      } else {
        throw new Error(data.message || 'Unknown error occurred');
      }

    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now() + 1,
        text: `Sorry, there was an error processing your request: ${error.message}. Please try again.`,
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = async () => {
    try {
      // CORRECTED: Use /api/history/clear
      await fetch(`${API_BASE_URL}/api/history/clear`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      setMessages([]);
      addSystemMessage('🗑️ Chat history cleared.');
      loadPerformanceStats(); // Refresh stats
    } catch (error) {
      console.error('Failed to clear chat:', error);
      setMessages([]);
    }
  };

  const reinitialize = async () => {
    // CORRECTED: No separate initialize endpoint needed - just check health
    setChatbotStatus('needs-init');
    addSystemMessage('🔄 Checking chatbot status...');
    await checkHealth();
    await loadDatabaseInfo();
    await loadPerformanceStats();
  };

  const getStatusColor = () => {
    switch (chatbotStatus) {
      case 'ready': return '#4CAF50';
      case 'needs-init': return '#FF9800';
      case 'error': return '#F44336';
      default: return '#9E9E9E';
    }
  };

  const getStatusText = () => {
    switch (chatbotStatus) {
      case 'ready': return 'Ready';
      case 'needs-init': return 'Initializing...';
      case 'error': return 'Error';
      default: return 'Disconnected';
    }
  };

  const formatResponseTime = (ms) => {
    if (!ms) return '';
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>📚 Educational RAG Chatbot</h1>
        <div className="status-indicator">
          <div 
            className="status-dot" 
            style={{ backgroundColor: getStatusColor() }}
          ></div>
          <span>{getStatusText()}</span>
        </div>
      </header>

      
      {dbInfo && (
        <div className="database-info">
          <div className="info-row">
            <strong>Knowledge Base:</strong> {dbInfo.total_chunks} chunks • {dbInfo.embedding_model}
          </div>
          {performanceStats && (
            <div className="info-row">
              <strong>Performance:</strong> {performanceStats.total_queries} queries • 
              Avg: {formatResponseTime(performanceStats.avg_response_time_ms)} • 
              Quality: {performanceStats.quality_grade} • Speed: {performanceStats.speed_grade}
            </div>
          )}
        </div>
      )}

      <div className="chat-container">
        <div className="messages">
          {messages.length === 0 && (
            <div className="welcome-message">
              <h3>🎓 Welcome to the Quality-Optimized Educational RAG Chatbot!</h3>
              <p>Ask me detailed questions about your educational materials.</p>
              <p>I'll provide comprehensive, relevant answers using advanced AI analysis.</p>
              <div className="features">
                <span className="feature">📚 Comprehensive Responses</span>
                <span className="feature">🎯 High Relevance</span>
                <span className="feature">⚡ 5-15s Response Time</span>
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`message ${message.sender} ${message.isError ? 'error' : ''}`}>
              <div className="message-content">
                <p>{message.text}</p>
                
                {/* ENHANCED: Better metadata display *//*)
                {message.confidence && (
                  <div className="message-meta">
                    <span className={`confidence confidence-${message.confidence}`}>
                      {message.confidence.toUpperCase()}
                    </span>
                    
                    {/* CORRECTED: Show sources count properly *//*}
                    <span className="sources-count">
                      📚 {message.sources_count || 0} sources
                    </span>
                    
                    {/* NEW: Show response time *//*}
                    {message.response_time && (
                      <span className="response-time">
                        ⚡ {formatResponseTime(message.response_time)}
                      </span>
                    )}
                    
                    {/* NEW: Show relevance score *//*}
                    {message.relevance_score && (
                      <span className="relevance-score">
                        🎯 {(message.relevance_score * 100).toFixed(0)}% relevant
                      </span>
                    )}
                  </div>
                )}
                <span className="timestamp">{message.timestamp}</span>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message bot loading">
              <div className="message-content">
                <div className="typing-indicator">
                  <div></div><div></div><div></div>
                </div>
                <p>Generating comprehensive response...</p>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSendMessage} className="input-form">
          <div className="input-container">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder={
                chatbotStatus === 'ready' 
                  ? "Ask a detailed question about your educational materials..." 
                  : "Please wait for chatbot to initialize..."
              }
              disabled={isLoading || chatbotStatus !== 'ready'}
              className="message-input"
            />
            <button 
              type="submit" 
              disabled={isLoading || chatbotStatus !== 'ready' || !inputMessage.trim()}
              className="send-button"
            >
              {isLoading ? '⏳' : '📤'}
            </button>
          </div>
        </form>

        <div className="chat-controls">
          <button onClick={clearChat} className="clear-button">
            🗑️ Clear Chat
          </button>
          <button onClick={reinitialize} className="reinit-button">
            🔄 Refresh Status
          </button>
          <button onClick={loadPerformanceStats} className="stats-button">
            📊 Update Stats
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;*/