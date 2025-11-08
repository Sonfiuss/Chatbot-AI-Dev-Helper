/**
 * Advanced AI Development Expert - Workshop 2 JavaScript
 * Enhanced chatbot với function calling, conversation management
 */

class AdvancedChatbot {
    constructor() {
        this.sessionId = null;
        this.conversationHistory = [];
        this.isProcessing = false;
        this.currentMode = 'advanced_chatbot';
        this.useFunctions = true;
        this.mockData = null;
        
          // === NEW: TTS toggle + audio player ===
        this.voiceEnabled = false;     // ALT+V để bật/tắt TTS
        this.audio = new Audio();

        this.init();
        this.loadMockData();
    }

    async init() {
        this.bindEvents();
        await this.checkStatus();
      // === NEW: seed KB vào Chroma (1 lần khi mở trang) ===
        await this.seedKB();
        this.setupAutoResize();
        this.initializeChat();
    }

    bindEvents() {
        // Chat input events
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });


        
        messageInput.addEventListener('input', this.updateCharCount.bind(this));
        sendButton.addEventListener('click', this.sendMessage.bind(this));

        // Header controls
        document.getElementById('chatMode').addEventListener('change', this.changeMode.bind(this));
        document.getElementById('functionsToggle').addEventListener('change', this.toggleFunctions.bind(this));
        document.getElementById('exportBtn').addEventListener('click', this.exportConversation.bind(this));
        document.getElementById('clearBtn').addEventListener('click', this.clearConversation.bind(this));
        document.getElementById('settingsBtn').addEventListener('click', this.showSettings.bind(this));

        // Quick actions
        document.querySelectorAll('.quick-action-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const action = btn.dataset.action;
                const prompt = btn.dataset.prompt;
                this.handleQuickAction(action, prompt);
            });
        });

        // Mock problems
        document.querySelectorAll('.mock-problem-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const problem = btn.dataset.problem;
                this.loadMockProblem(problem);
            });
        });

        // Code input actions
        document.getElementById('attachCodeBtn').addEventListener('click', this.attachCode.bind(this));
        document.getElementById('pasteCodeBtn').addEventListener('click', this.pasteFromClipboard.bind(this));
        document.getElementById('clearCodeBtn').addEventListener('click', this.clearCode.bind(this));

        // Suggestion chips
        document.querySelectorAll('.suggestion-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                document.getElementById('messageInput').value = chip.dataset.text;
                this.updateCharCount();
                document.getElementById('messageInput').focus();
            });
        });

        // Settings modal
        document.getElementById('closeSettings').addEventListener('click', this.hideSettings.bind(this));
        
        // Click outside modal to close
        document.getElementById('settingsModal').addEventListener('click', (e) => {
            if (e.target === e.currentTarget) {
                this.hideSettings();
            }
        });
        // === NEW: Alt+V để bật/tắt voice (TTS) ===
        document.addEventListener('keydown', (e) => {
        if (e.altKey && e.key.toLowerCase() === 'v') {
            this.voiceEnabled = !this.voiceEnabled;
            this.showToast(`Voice ${this.voiceEnabled ? 'ON' : 'OFF'}`, 'success');
        }
        });
    }

    async loadMockData() {
        try {
            const response = await fetch('/static/mock_data.json');
            if (response.ok) {
                this.mockData = await response.json();
            }
        } catch (error) {
            console.warn('Could not load mock data:', error);
        }
    }

    // === NEW: Seed knowledge base vào Chroma từ static/mock_data.json ===
    async seedKB() {
    try {
        const res = await fetch('/api/kb/seed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
        });
        const data = await res.json();
        if (data.success) {
        console.log(`[KB] Seeded ${data.count} items`);
        } else {
        console.warn('[KB] seed fail:', data.error);
        }
    } catch (e) {
        console.warn('[KB] seed error:', e);
    }
    }


    setupAutoResize() {
        const messageInput = document.getElementById('messageInput');
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
    }

    initializeChat() {
        this.updateChatTitle();
        this.scrollToBottom();
    }

    async checkStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            this.updateStatusIndicator(status);
            
            if (!status.api_connected) {
                this.showToast('API connection issue. Some features may not work.', 'warning');
            }
        } catch (error) {
            console.error('Status check failed:', error);
            this.updateStatusIndicator({ status: 'error', api_connected: false });
        }
    }

    updateStatusIndicator(status) {
        const indicator = document.getElementById('statusIndicator');
        const dot = indicator.querySelector('.status-dot');
        const text = indicator.querySelector('.status-text');
        
        if (status.api_connected) {
            dot.className = 'status-dot';
            text.textContent = `Connected (${status.client_type || 'API'})`;
        } else {
            dot.className = 'status-dot error';
            text.textContent = 'Connection Error';
        }
    }

    changeMode(event) {
        this.currentMode = event.target.value;
        this.updateChatTitle();
        
        const modeDescriptions = {
            'advanced_chatbot': 'Multi-turn conversation với function calling và advanced prompting',
            'code_reviewer': 'Detailed code analysis với security, performance, và best practices review',
            'debugging_assistant': 'Step-by-step debugging help với systematic problem solving'
        };
        
        document.getElementById('chatSubtitle').textContent = modeDescriptions[this.currentMode];
        
        this.showToast(`Switched to ${event.target.selectedOptions[0].textContent}`, 'success');
    }

    updateChatTitle() {
        const titles = {
            'advanced_chatbot': 'Advanced AI Development Assistant',
            'code_reviewer': 'Senior Code Reviewer',
            'debugging_assistant': 'Debugging Expert'
        };
        
        document.getElementById('chatTitle').textContent = titles[this.currentMode];
    }

    toggleFunctions(event) {
        this.useFunctions = event.target.checked;
        this.showToast(`Function calling ${this.useFunctions ? 'enabled' : 'disabled'}`, 'success');
    }

    async sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();
        
        if (!message || this.isProcessing) return;
        
        this.isProcessing = true;
        this.updateSendButton(true);
        this.showTypingIndicator(true);
        
        // Add user message to chat
        this.addMessage('user', message);
        messageInput.value = '';
        this.updateCharCount();
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    mode: this.currentMode,
                    use_functions: this.useFunctions,
                    history: this.conversationHistory,

                    // === NEW: yêu cầu server trả TTS khi bật ===
                    speak: this.voiceEnabled
                })

            });
            
            const data = await response.json();
            
            if (data.success) {
                let botContent = data.response || '';
                if (data.retrieval && Array.isArray(data.retrieval) && data.retrieval.length) {
                const names = data.retrieval
                    .map(r => r?.metadata?.title || r?.metadata?.genre || r?.metadata?.source || r?.id)
                    .filter(Boolean).join(' · ');
                botContent += `\n\nKB: ${names}`;
                }

                // dùng botContent thay cho data.response
                this.addMessage('assistant', botContent, {
                function_called: data.function_called,
                function_result: data.function_result
                });

                // Add bot response
                // this.addMessage('assistant', data.response, {
                //     function_called: data.function_called,
                //     function_result: data.function_result
                // });
                // === NEW: phát audio nếu server trả về audio_base64 ===
                
                if (this.voiceEnabled && data.audio_base64) {
                this.audio.src = `data:audio/wav;base64,${data.audio_base64}`;
                try { await this.audio.play(); } catch (err) { console.warn('Audio play error', err); }
                }

                // Update conversation context
                if (data.conversation_context) {
                    this.updateConversationContext(data.conversation_context);
                }
                
                this.conversationHistory.push(
                    { role: 'user', content: message },
                    { role: 'assistant', content: data.response }
                );
            } else {
                this.addMessage('assistant', `❌ Error: ${data.error}`, { isError: true });
            }
        } catch (error) {
            console.error('Chat error:', error);
            this.addMessage('assistant', '❌ Connection error. Please try again.', { isError: true });
        } finally {
            this.isProcessing = false;
            this.updateSendButton(false);
            this.showTypingIndicator(false);
        }
    }

    addMessage(role, content, metadata = {}) {
        const messagesContainer = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;
        
        const timestamp = new Date().toLocaleTimeString('vi-VN', {
            hour: '2-digit',
            minute: '2-digit'
        });
        
        const senderName = role === 'user' ? 'You' : this.getSenderName();
        const avatarIcon = role === 'user' ? 'fas fa-user' : 'fas fa-brain';
        
        let functionCallHtml = '';
        if (metadata.function_called) {
            functionCallHtml = `
                <div class="function-call-display">
                    <div class="function-call-header">
                        <i class="fas fa-cog"></i>
                        <span class="function-call-name">Function: ${metadata.function_called}</span>
                    </div>
                    <div class="function-call-args">
                        ${this.formatFunctionResult(metadata.function_result)}
                    </div>
                </div>
            `;
        }
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="${avatarIcon}"></i>
            </div>
            <div class="message-content">
                <div class="message-bubble ${metadata.isError ? 'error' : ''}">
                    <div class="message-header">
                        <span class="sender-name">${senderName}</span>
                        <span class="message-time">${timestamp}</span>
                    </div>
                    <div class="message-text">
                        ${this.formatMessage(content)}
                    </div>
                    ${functionCallHtml}
                </div>
            </div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Highlight code blocks
        this.highlightCode(messageDiv);
    }

    getSenderName() {
        const names = {
            'advanced_chatbot': 'AI Development Expert',
            'code_reviewer': 'Senior Code Reviewer',
            'debugging_assistant': 'Debugging Expert'
        };
        return names[this.currentMode];
    }

    formatMessage(content) {
        // Convert markdown-like formatting to HTML
        content = content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>');
        
        // Handle code blocks
        content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            const language = lang || 'python';
            return `<pre class="language-${language}"><code>${this.escapeHtml(code.trim())}</code></pre>`;
        });
        
        // Handle lists
        content = content.replace(/^\d+\.\s(.+)/gm, '<li>$1</li>');
        content = content.replace(/^[-•]\s(.+)/gm, '<li>$1</li>');
        content = content.replace(/(<li>.*<\/li>)/s, '<ol>$1</ol>');
        
        return `<p>${content}</p>`;
    }

    formatFunctionResult(result) {
        if (!result) return '';
        
        if (typeof result === 'object') {
            return `<pre>${JSON.stringify(result, null, 2)}</pre>`;
        }
        
        return this.escapeHtml(String(result));
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    highlightCode(element) {
        if (window.Prism) {
            const codeBlocks = element.querySelectorAll('pre code');
            codeBlocks.forEach(block => {
                Prism.highlightElement(block);
            });
        }
    }

    updateConversationContext(context) {
        document.getElementById('messageCount').textContent = context.message_count || 0;
        document.getElementById('functionCount').textContent = context.function_calls_used || 0;
        
        const topicTags = document.getElementById('topicTags');
        topicTags.innerHTML = '';
        
        if (context.topics_discussed && context.topics_discussed.length > 0) {
            context.topics_discussed.forEach(topic => {
                const tag = document.createElement('span');
                tag.className = 'topic-tag';
                tag.textContent = topic;
                topicTags.appendChild(tag);
            });
        }
    }

    updateCharCount() {
        const messageInput = document.getElementById('messageInput');
        const charCount = document.getElementById('charCount');
        charCount.textContent = messageInput.value.length;
    }

    updateSendButton(disabled) {
        const sendButton = document.getElementById('sendButton');
        sendButton.disabled = disabled;
    }

    showTypingIndicator(show) {
        const indicator = document.getElementById('typingIndicator');
        indicator.style.display = show ? 'flex' : 'none';
    }

    scrollToBottom() {
        const messagesContainer = document.getElementById('chatMessages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    handleQuickAction(action, prompt) {
        const codeInput = document.getElementById('codeInput');
        const code = codeInput.value.trim();
        
        let fullPrompt = prompt;
        if (code && action !== 'solution' && action !== 'architecture') {
            fullPrompt += `\n\n\`\`\`python\n${code}\n\`\`\``;
        }
        
        document.getElementById('messageInput').value = fullPrompt;
        this.updateCharCount();
        document.getElementById('messageInput').focus();
        
        // Auto-send for certain actions
        if (['analyze', 'debug'].includes(action) && code) {
            setTimeout(() => this.sendMessage(), 100);
        }
    }

    loadMockProblem(problemId) {
        if (!this.mockData) {
            this.showToast('Mock data not available', 'warning');
            return;
        }
        
        const problem = this.mockData.company_problems.find(p => p.id === `problem_${problemId.split('_')[0]}`);
        if (problem) {
            const prompt = `Tôi cần giải pháp cho vấn đề: ${problem.title}

**Mô tả:** ${problem.description}

**Context:**
${Object.entries(problem.context).map(([key, value]) => `- ${key}: ${Array.isArray(value) ? value.join(', ') : value}`).join('\n')}

**Requirements:**
${problem.requirements.map(req => `- ${req}`).join('\n')}

Hãy phân tích và đưa ra giải pháp chi tiết.`;

            document.getElementById('messageInput').value = prompt;
            this.updateCharCount();
            document.getElementById('messageInput').focus();
        }
    }

    attachCode() {
        const codeInput = document.getElementById('codeInput');
        const messageInput = document.getElementById('messageInput');
        const code = codeInput.value.trim();
        
        if (code) {
            const currentMessage = messageInput.value;
            const newMessage = currentMessage + (currentMessage ? '\n\n' : '') + `\`\`\`python\n${code}\n\`\`\``;
            messageInput.value = newMessage;
            this.updateCharCount();
            messageInput.focus();
            this.showToast('Code attached to message', 'success');
        } else {
            this.showToast('No code to attach', 'warning');
        }
    }

    async pasteFromClipboard() {
        try {
            const text = await navigator.clipboard.readText();
            document.getElementById('codeInput').value = text;
            this.showToast('Code pasted from clipboard', 'success');
        } catch (error) {
            this.showToast('Could not access clipboard', 'error');
        }
    }

    clearCode() {
        document.getElementById('codeInput').value = '';
        this.showToast('Code cleared', 'success');
    }

    async exportConversation() {
        try {
            this.showLoading(true);
            
            const response = await fetch('/api/conversation/context');
            const data = await response.json();
            
            if (data.success && data.context.session_id) {
                const exportResponse = await fetch(`/api/conversation/export/${data.context.session_id}`);
                const exportData = await exportResponse.json();
                
                if (exportData.success) {
                    this.downloadJSON(exportData.conversation, 'conversation_export.json');
                    this.showToast('Conversation exported successfully', 'success');
                } else {
                    this.showToast('Export failed', 'error');
                }
            } else {
                this.showToast('No conversation to export', 'warning');
            }
        } catch (error) {
            console.error('Export error:', error);
            this.showToast('Export failed', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    downloadJSON(data, filename) {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }

    async clearConversation() {
        if (confirm('Are you sure you want to clear the conversation?')) {
            try {
                const response = await fetch('/api/conversation/clear');
                const data = await response.json();
                
                if (data.success) {
                    // Clear UI
                    const messagesContainer = document.getElementById('chatMessages');
                    const firstMessage = messagesContainer.querySelector('.message');
                    messagesContainer.innerHTML = '';
                    messagesContainer.appendChild(firstMessage); // Keep welcome message
                    
                    // Reset state
                    this.conversationHistory = [];
                    this.updateConversationContext({ message_count: 0, function_calls_used: 0, topics_discussed: [] });
                    
                    this.showToast('Conversation cleared', 'success');
                } else {
                    this.showToast('Clear failed', 'error');
                }
            } catch (error) {
                console.error('Clear error:', error);
                this.showToast('Clear failed', 'error');
            }
        }
    }

    showSettings() {
        document.getElementById('settingsModal').style.display = 'block';
    }

    hideSettings() {
        document.getElementById('settingsModal').style.display = 'none';
    }

    showLoading(show) {
        document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
    }

    showToast(message, type = 'success') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        
        container.appendChild(toast);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            toast.remove();
        }, 3000);
    }
}

// Initialize chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatbot = new AdvancedChatbot();
});