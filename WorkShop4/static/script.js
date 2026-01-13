// Workshop 4 - Code Assistant Chatbot JavaScript

// State management
let chatHistory = [];
let isProcessing = false;

// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const statusIndicator = document.getElementById('status');
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    setupEventListeners();
    setupFileUpload();
});

// Setup event listeners
function setupEventListeners() {
    sendBtn.addEventListener('click', handleSendMessage);
    clearBtn.addEventListener('click', handleClearChat);
    
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    // Example queries
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            userInput.value = btn.dataset.query;
            userInput.focus();
        });
    });
}

// Setup file upload
function setupFileUpload() {
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

// Handle file upload
async function handleFileUpload(file) {
    if (!file) return;

    const allowedExtensions = ['py', 'txt', 'js', 'java', 'cpp', 'c', 'json'];
    const fileExt = file.name.split('.').pop().toLowerCase();

    if (!allowedExtensions.includes(fileExt)) {
        showNotification('Chỉ chấp nhận file: ' + allowedExtensions.join(', '), 'error');
        return;
    }

    if (file.size > 5 * 1024 * 1024) {
        showNotification('File quá lớn! Tối đa 5MB', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        showNotification('Đang upload và analyze file...', 'info');
        
        const response = await fetch('/api/upload-code', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            // Display uploaded file content
            addMessage('user', `📎 Uploaded file: ${data.filename}\n\`\`\`python\n${data.code.substring(0, 500)}${data.code.length > 500 ? '\n...' : ''}\n\`\`\``);
            
            // Display analysis results
            const analysis = JSON.parse(data.analysis);
            let analysisMessage = `📊 **Code Analysis Results:**\n\n`;
            
            if (analysis.status === 'syntax_error') {
                analysisMessage += `🔴 **Syntax Error:**\n${analysis.error}\n\nSeverity: ${analysis.severity}`;
            } else {
                analysisMessage += `✅ Analysis completed!\n\n`;
                analysisMessage += `📈 Issues found: ${analysis.issues_found}\n\n`;
                
                if (analysis.issues && analysis.issues.length > 0) {
                    analysisMessage += `**Issues detected:**\n`;
                    analysis.issues.forEach((issue, idx) => {
                        const icon = issue.severity === 'CRITICAL' ? '🔴' : issue.severity === 'HIGH' ? '🟡' : '🔵';
                        analysisMessage += `${idx + 1}. ${icon} **${issue.type}** (${issue.severity}): ${issue.description}\n`;
                    });
                } else {
                    analysisMessage += `✨ No issues detected! Code looks good.\n`;
                }
                
                analysisMessage += `\n💡 **Recommendation:** ${analysis.recommendation}`;
            }
            
            addMessage('bot', analysisMessage);
            
            showNotification('File analyzed successfully!', 'success');
            fileInput.value = '';
        } else {
            showNotification(data.error, 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showNotification('Lỗi upload file: ' + error.message, 'error');
    }
}

// Check system status
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (data.status === 'running' && data.rag_ready) {
            updateStatus('online', 'Online - RAG Ready');
        } else {
            updateStatus('warning', 'System starting...');
        }
    } catch (error) {
        updateStatus('offline', 'Connection error');
    }
}

// Update status indicator
function updateStatus(status, text) {
    const statusMap = {
        'online': { color: '#10b981', icon: 'fa-circle' },
        'warning': { color: '#f59e0b', icon: 'fa-circle' },
        'offline': { color: '#ef4444', icon: 'fa-circle' }
    };
    
    const config = statusMap[status] || statusMap.offline;
    statusIndicator.querySelector('i').style.color = config.color;
    statusIndicator.querySelector('span').textContent = text;
}

// Handle send message
async function handleSendMessage() {
    const message = userInput.value.trim();
    
    if (!message || isProcessing) return;
    
    // Add user message
    addMessage('user', message);
    userInput.value = '';
    userInput.style.height = 'auto';
    
    isProcessing = true;
    sendBtn.disabled = true;
    sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Add bot response
            addMessage('bot', data.response);
            
            // Show sources if available
            if (data.sources && data.sources.length > 0) {
                addSources(data.sources);
            }
            
            chatHistory.push({ user: message, bot: data.response });
        } else {
            addMessage('bot', `❌ Error: ${data.error}`);
        }
    } catch (error) {
        console.error('Chat error:', error);
        addMessage('bot', `❌ Lỗi kết nối: ${error.message}`);
    } finally {
        isProcessing = false;
        sendBtn.disabled = false;
        sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i> Send';
    }
}

// Add message to chat
function addMessage(sender, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender === 'user' ? 'user-message' : 'bot-message'}`;
    
    const avatarDiv = document.createElement('div');
    avatarDiv.className = `message-avatar ${sender === 'user' ? 'user-avatar' : 'bot-avatar'}`;
    avatarDiv.innerHTML = sender === 'user' 
        ? '<i class="fas fa-user"></i>' 
        : '<i class="fas fa-robot"></i>';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Format content with markdown-like syntax
    const formattedContent = formatMessage(content);
    contentDiv.innerHTML = formattedContent;
    
    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Format message with markdown-like syntax
function formatMessage(text) {
    // Code blocks
    text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang || 'text'}">${escapeHtml(code.trim())}</code></pre>`;
    });
    
    // Inline code
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Bold
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Lists
    text = text.replace(/^- (.+)$/gm, '<li>$1</li>');
    text = text.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
    
    // Numbered lists
    text = text.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
    
    // Paragraphs
    text = text.split('\n\n').map(para => {
        if (!para.match(/^<(ul|ol|pre|code)/)) {
            return `<p>${para}</p>`;
        }
        return para;
    }).join('\n');
    
    // Line breaks
    text = text.replace(/\n/g, '<br>');
    
    return text;
}

// Escape HTML
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Add sources section
function addSources(sources) {
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'sources-section';
    
    let sourcesHtml = '<div class="sources-header"><i class="fas fa-book"></i> Knowledge Base Sources:</div>';
    sourcesHtml += '<div class="sources-list">';
    
    sources.forEach((source, idx) => {
        const severityIcon = source.severity === 'critical' ? '🔴' : 
                           source.severity === 'high' ? '🟡' : 
                           source.severity === 'medium' ? '🟠' : '🔵';
        
        sourcesHtml += `
            <div class="source-item">
                <div class="source-header">
                    <strong>${severityIcon} ${source.source}</strong>
                    ${source.severity ? `<span class="severity-badge ${source.severity}">${source.severity}</span>` : ''}
                </div>
                <div class="source-content">${source.content}</div>
                ${source.example ? `<details><summary>Example Code</summary><pre><code>${escapeHtml(source.example)}</code></pre></details>` : ''}
            </div>
        `;
    });
    
    sourcesHtml += '</div>';
    sourcesDiv.innerHTML = sourcesHtml;
    
    chatMessages.appendChild(sourcesDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Clear chat
async function handleClearChat() {
    if (!confirm('Xóa toàn bộ chat history?')) return;
    
    try {
        const response = await fetch('/api/clear-history', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            chatMessages.innerHTML = '';
            chatHistory = [];
            showNotification('Chat history cleared!', 'success');
            
            // Add welcome message again
            addMessage('bot', '🤖 Chat history đã được xóa. Hãy bắt đầu cuộc trò chuyện mới!');
        }
    } catch (error) {
        showNotification('Lỗi khi xóa history: ' + error.message, 'error');
    }
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    const icon = type === 'success' ? 'fa-check-circle' : 
                 type === 'error' ? 'fa-exclamation-circle' : 
                 'fa-info-circle';
    
    notification.innerHTML = `
        <i class="fas ${icon}"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Auto-resize textarea
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
});
