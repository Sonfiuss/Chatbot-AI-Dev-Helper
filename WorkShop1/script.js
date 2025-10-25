// Các biến global
let chatHistory = [];

// Khởi tạo khi trang web load xong
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
});

function initializeApp() {
    // Hiển thị thời gian cho tin nhắn bot đầu tiên
    const firstMessage = document.querySelector('.message-time');
    if (firstMessage) {
        firstMessage.textContent = formatTime(new Date());
    }
    
    // Auto-focus vào method input
    const methodInput = document.getElementById('methodInput');
    if (methodInput) {
        methodInput.focus();
    }
}

function setupEventListeners() {
    // Method input actions
    const clearBtn = document.getElementById('clearBtn');
    const copyBtn = document.getElementById('copyBtn');
    const methodInput = document.getElementById('methodInput');
    
    clearBtn.addEventListener('click', function() {
        if (confirm('Xóa toàn bộ code đã nhập?')) {
            methodInput.value = '';
            methodInput.focus();
        }
    });
    
    copyBtn.addEventListener('click', function() {
        if (methodInput.value.trim()) {
            navigator.clipboard.writeText(methodInput.value).then(() => {
                showToast('Đã copy code!', 'success');
            }).catch(() => {
                showToast('Không thể copy code', 'error');
            });
        } else {
            showToast('Không có code để copy', 'warning');
        }
    });

    // Quick action buttons
    const quickBtns = document.querySelectorAll('.quick-btn');
    quickBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const action = this.dataset.action;
            handleQuickAction(action);
        });
    });

    // Gửi tin nhắn
    const sendBtn = document.getElementById('sendButton');
    const messageInput = document.getElementById('messageInput');
    
    sendBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Gợi ý nhanh
    const hints = document.querySelectorAll('.hint');
    hints.forEach(hint => {
        hint.addEventListener('click', function() {
            const text = this.textContent.replace(/[💡�⚡🧪]/g, '').trim();
            messageInput.value = getHintMessage(text);
            messageInput.focus();
        });
    });
    
    // Tab support for method input
    methodInput.addEventListener('keydown', function(e) {
        if (e.key === 'Tab') {
            e.preventDefault();
            const start = this.selectionStart;
            const end = this.selectionEnd;
            
            // Insert 4 spaces
            this.value = this.value.substring(0, start) + '    ' + this.value.substring(end);
            this.selectionStart = this.selectionEnd = start + 4;
        }
    });
}

function handleQuickAction(action) {
    const methodInput = document.getElementById('methodInput');
    const code = methodInput.value.trim();
    
    if (!code) {
        showToast('Hãy nhập code Python trước!', 'warning');
        methodInput.focus();
        return;
    }
    
    const messageInput = document.getElementById('messageInput');
    let message = '';
    
    switch(action) {
        case 'explain':
            message = 'Giải thích cho tôi đoạn code Python này làm gì:';
            break;
        case 'optimize':
            message = 'Hãy tối ưu hóa đoạn code Python này:';
            break;
        case 'debug':
            message = 'Tìm lỗi và sửa đoạn code Python này:';
            break;
        case 'test':
            message = 'Viết unit test cho đoạn code Python này:';
            break;
    }
    
    messageInput.value = message;
    messageInput.focus();
}

function getHintMessage(hint) {
    const hintMessages = {
        'Giải thích code này': 'Giải thích cho tôi code này hoạt động như thế nào?',
        'Cách cải thiện?': 'Code này có thể cải thiện như thế nào?',
        'Tối ưu performance': 'Làm sao để tối ưu performance của code này?',
        'Viết unit test': 'Viết unit test cho code này'
    };
    return hintMessages[hint] || hint;
}

function showToast(message, type = 'info') {
    // Tạo toast notification
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <i class="fas ${getToastIcon(type)}"></i>
        <span>${message}</span>
    `;
    
    // Style cho toast
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--bg-tertiary);
        color: var(--text-primary);
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 4px solid ${getToastColor(type)};
        box-shadow: var(--shadow-md);
        z-index: 1000;
        animation: slideInRight 0.3s ease;
        display: flex;
        align-items: center;
        gap: 8px;
        max-width: 300px;
    `;
    
    document.body.appendChild(toast);
    
    // Tự động xóa sau 3 giây
    setTimeout(() => {
        toast.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 300);
    }, 3000);
}

function getToastIcon(type) {
    const icons = {
        'success': 'fa-check-circle',
        'error': 'fa-times-circle',
        'warning': 'fa-exclamation-triangle',
        'info': 'fa-info-circle'
    };
    return icons[type] || icons['info'];
}

function getToastColor(type) {
    const colors = {
        'success': 'var(--accent-success)',
        'error': 'var(--accent-warning)',
        'warning': '#fbbf24',
        'info': 'var(--accent-primary)'
    };
    return colors[type] || colors['info'];
}

async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const methodInput = document.getElementById('methodInput');
    const message = messageInput.value.trim();
    
    if (!message) return;
    
    // Lấy code từ method input nếu có
    const code = methodInput.value.trim();
    let fullMessage = message;
    
    if (code) {
        fullMessage = `${message}\n\n\`\`\`python\n${code}\n\`\`\``;
    }
    
    // Hiển thị tin nhắn user
    addMessage('user', message);
    messageInput.value = '';
    
    // Hiển thị typing indicator
    const typingId = addTypingIndicator();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: fullMessage,
                history: chatHistory
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            removeTypingIndicator(typingId);
            addMessage('bot', result.response);
            
            // Cập nhật lịch sử chat
            chatHistory.push(
                { role: 'user', content: fullMessage },
                { role: 'assistant', content: result.response }
            );
        } else {
            removeTypingIndicator(typingId);
            addMessage('system', '❌ Lỗi: ' + (result.error || 'Không thể gửi tin nhắn'));
        }
    } catch (error) {
        console.error('Send message error:', error);
        removeTypingIndicator(typingId);
        addMessage('system', '❌ Lỗi kết nối server. Hãy đảm bảo server Flask đang chạy.');
    }
}

function addMessage(sender, content) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageElement = document.createElement('div');
    
    const isUser = sender === 'user';
    const isSystem = sender === 'system';
    
    messageElement.className = `message ${isUser ? 'user-message' : isSystem ? 'system-message' : 'bot-message'}`;
    
    if (isSystem) {
        messageElement.innerHTML = `
            <div class="message-content" style="max-width: 100%; text-align: center;">
                <div class="message-bubble" style="background: var(--bg-hover); border-color: var(--accent-primary); border-left: 4px solid var(--accent-primary);">
                    <p>${content}</p>
                </div>
                <div class="message-time">${formatTime(new Date())}</div>
            </div>
        `;
    } else {
        const avatarIcon = isUser ? 'fa-user' : 'fa-robot';
        const avatarClass = isUser ? 'user-avatar' : 'bot-avatar';
        
        messageElement.innerHTML = `
            <div class="message-avatar ${avatarClass}">
                <i class="fas ${avatarIcon}"></i>
            </div>
            <div class="message-content">
                <div class="message-bubble">
                    ${formatMessage(content)}
                </div>
                <div class="message-time">${formatTime(new Date())}</div>
            </div>
        `;
    }
    
    messagesContainer.appendChild(messageElement);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function addTypingIndicator() {
    const messagesContainer = document.getElementById('chatMessages');
    const typingElement = document.createElement('div');
    const typingId = 'typing-' + Date.now();
    
    typingElement.id = typingId;
    typingElement.className = 'message bot-message';
    typingElement.innerHTML = `
        <div class="message-avatar bot-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <div class="message-bubble" style="background: #f8f9fa;">
                <div style="display: flex; align-items: center; gap: 4px;">
                    <span>Đang trả lời</span>
                    <div class="typing-dots">
                        <span style="animation: typing 1.4s infinite ease-in-out;">.</span>
                        <span style="animation: typing 1.4s infinite ease-in-out 0.2s;">.</span>
                        <span style="animation: typing 1.4s infinite ease-in-out 0.4s;">.</span>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    messagesContainer.appendChild(typingElement);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return typingId;
}

function removeTypingIndicator(typingId) {
    const typingElement = document.getElementById(typingId);
    if (typingElement) {
        typingElement.remove();
    }
}

function formatMessage(message) {
    // Xử lý markdown đơn giản với night theme
    let formatted = message
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
    
    // Xử lý code blocks
    formatted = formatted.replace(/```python\n([\s\S]*?)\n```/g, function(match, code) {
        return `<pre style="background: var(--bg-primary); padding: 12px; border-radius: 6px; overflow-x: auto; margin: 8px 0; border-left: 3px solid var(--python-blue);"><code style="color: var(--text-primary); background: none; padding: 0;">${code.trim()}</code></pre>`;
    });
    
    // Xử lý bullet points
    formatted = formatted.replace(/^• (.*$)/gim, '<div style="margin: 4px 0;">• $1</div>');
    
    // Wrap trong thẻ p nếu không có block elements
    if (!formatted.includes('<pre>') && !formatted.includes('<div>')) {
        formatted = `<p>${formatted}</p>`;
    }
    
    return formatted;
}

function formatTime(date) {
    return date.toLocaleTimeString('vi-VN', {
        hour: '2-digit',
        minute: '2-digit'
    });
}

// CSS cho animations
const style = document.createElement('style');
style.textContent = `
    @keyframes typing {
        0%, 60%, 100% { opacity: 0.4; }
        30% { opacity: 1; }
    }
    
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .typing-dots span {
        font-size: 14px;
        font-weight: bold;
        color: var(--accent-primary);
    }
`;
document.head.appendChild(style);