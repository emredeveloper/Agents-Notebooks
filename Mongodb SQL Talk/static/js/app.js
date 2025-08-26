// MongoDB LangChain Agent - Enhanced Modern JavaScript

class MongoDBAgent {
    constructor() {
        this.isProcessing = false;
        this.messageCount = 0;
        this.autoScroll = true;
        this.selectedCollection = null;
        this.collections = [];
        this.initializeApp();
    }

    initializeApp() {
        // Set initial status
        this.updateStatus('ready');
        
        // Focus on input field
        document.getElementById('queryInput').focus();
        
        // Initialize message counter
        this.updateMessageCount();
        
        // Setup input suggestions
        this.setupInputSuggestions();
        
        // Load collections
        this.loadCollections();
        
        console.log('🚀 MongoDB Agent initialized with modern UI');
    }

    setupInputSuggestions() {
        const input = document.getElementById('queryInput');
        const suggestions = [
            'koleksiyonları listele',
            'kaç collection var',
            'ilk 10 veriyi göster',
            'adı Ahmet olanları bul',
            'yaşı 25den büyük olanlar',
            'yeni kullanıcı ekle',
            'sütunları ingilizce güncelle',
            'fiyatı 100den fazla ürünler'
        ];
        
        input.addEventListener('input', (e) => {
            const value = e.target.value.toLowerCase();
            const suggestionsDiv = document.getElementById('suggestions');
            
            if (value.length > 2) {
                const matches = suggestions.filter(s => s.includes(value));
                if (matches.length > 0) {
                    suggestionsDiv.innerHTML = matches.slice(0, 5).map(match => 
                        `<div class="suggestion-item" onclick="selectSuggestion('${match}')">${match}</div>`
                    ).join('');
                    suggestionsDiv.style.display = 'block';
                } else {
                    suggestionsDiv.style.display = 'none';
                }
            } else {
                suggestionsDiv.style.display = 'none';
            }
        });
        
        // Hide suggestions when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.input-field-wrapper')) {
                document.getElementById('suggestions').style.display = 'none';
            }
        });
    }

    updateMessageCount() {
        document.getElementById('messageCount').textContent = this.messageCount;
    }

    async loadCollections() {
        try {
            console.log('Loading collections...');
            const response = await fetch('/collections');
            console.log('Collections response status:', response.status);
            
            if (response.ok) {
                const data = await response.json();
                console.log('Collections data:', data);
                
                if (data.success && data.collections) {
                    this.collections = data.collections;
                    this.updateCollectionDropdown();
                    console.log('Collections loaded successfully:', this.collections.length);
                } else {
                    console.warn('No collections found or error in response:', data);
                    if (data.error) {
                        console.error('Collections error:', data.error);
                        // Show a user-friendly message
                        this.addMessage(`⚠️ Collection'lar yüklenemedi: ${data.error}. MongoDB bağlantısını kontrol edin.`, false);
                    }
                }
            } else {
                console.error('Collections endpoint error:', response.status);
            }
        } catch (error) {
            console.error('Collections yüklenemedi:', error);
        }
    }

    updateCollectionDropdown() {
        const select = document.getElementById('collectionSelect');
        console.log('Updating dropdown with collections:', this.collections);
        
        // Clear existing options except the first one
        while (select.children.length > 1) {
            select.removeChild(select.lastChild);
        }
        
        // Add collection options
        if (this.collections && this.collections.length > 0) {
            this.collections.forEach(collection => {
                const option = document.createElement('option');
                option.value = collection.name;
                option.textContent = `📊 ${collection.name} (${collection.count} kayıt)`;
                select.appendChild(option);
                console.log('Added collection option:', collection.name);
            });
        } else {
            console.warn('No collections to add to dropdown');
        }
    }

    onCollectionChange() {
        const select = document.getElementById('collectionSelect');
        const selectedValue = select.value;
        
        if (selectedValue) {
            this.selectedCollection = selectedValue;
            document.getElementById('activeContext').textContent = selectedValue;
            document.getElementById('activeContext').className = 'info-value collection';
            
            // Update record count
            const collection = this.collections.find(c => c.name === selectedValue);
            if (collection) {
                document.getElementById('recordCount').textContent = collection.count;
            }
            
            // Add context message
            this.addMessage(`🎯 Collection context '${selectedValue}' seçildi. Artık sorularınız bu collection ile ilişkili olacak.`, false);
        } else {
            this.selectedCollection = null;
            document.getElementById('activeContext').textContent = 'Database';
            document.getElementById('activeContext').className = 'info-value database';
            document.getElementById('recordCount').textContent = '-';
            
            // Add context message
            this.addMessage(`🗄️ Database context seçildi. Genel veritabanı sorguları yapabilirsiniz.`, false);
        }
        
        this.scrollToBottom();
    }

    async refreshCollections() {
        const refreshBtn = document.querySelector('.refresh-btn');
        refreshBtn.style.transform = 'rotate(360deg)';
        
        await this.loadCollections();
        
        setTimeout(() => {
            refreshBtn.style.transform = 'rotate(0deg)';
        }, 500);
    }

    updateStatus(status) {
        const statusElement = document.getElementById('status');
        statusElement.className = status;
        
        switch(status) {
            case 'ready':
                statusElement.textContent = 'Ready';
                break;
            case 'processing':
                statusElement.textContent = 'Processing...';
                break;
            case 'error':
                statusElement.textContent = 'Error';
                break;
        }
    }

    addMessage(content, isUser = false) {
        const chatContainer = document.getElementById('chatContainer');
        const messageDiv = document.createElement('div');
        const timestamp = new Date().toLocaleTimeString('tr-TR');
        
        messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        
        const avatarClass = isUser ? 'user-avatar' : 'bot-avatar';
        const icon = isUser ? 'fas fa-user' : 'fas fa-robot';
        const name = isUser ? 'Sen' : 'MongoDB Agent';
        
        messageDiv.innerHTML = `
            <div class="message-header">
                <div class="avatar ${avatarClass}">
                    <i class="${icon}"></i>
                </div>
                <div class="message-info">
                    <span class="sender">${name}</span>
                    <span class="timestamp">${timestamp}</span>
                </div>
            </div>
            <div class="message-content">
                ${content}
            </div>
        `;
        
        chatContainer.appendChild(messageDiv);
        
        // Update message count
        this.messageCount++;
        this.updateMessageCount();
        
        // Auto scroll if enabled
        if (this.autoScroll) {
            this.scrollToBottom();
        }
        
        // Add entrance animation
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            messageDiv.style.transition = 'all 0.4s ease-out';
            messageDiv.style.opacity = '1';
            messageDiv.style.transform = 'translateY(0)';
        }, 10);
    }

    scrollToBottom() {
        const chatContainer = document.getElementById('chatContainer');
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    formatData(data) {
        if (!data || data.length === 0) {
            return '<div style="text-align: center; color: #666; font-style: italic;">Veri bulunamadı</div>';
        }
        
        let html = '<table class="data-table"><thead><tr>';
        const keys = Object.keys(data[0]);
        
        // Filter out _id field and format headers
        const displayKeys = keys.filter(key => key !== '_id');
        displayKeys.forEach(key => {
            const formattedKey = this.formatFieldName(key);
            html += `<th>${formattedKey}</th>`;
        });
        html += '</tr></thead><tbody>';
        
        // Show max 10 records
        const displayData = data.slice(0, 10);
        displayData.forEach((item, index) => {
            html += `<tr style="animation-delay: ${index * 0.1}s">`;
            displayKeys.forEach(key => {
                let value = item[key] || '';
                
                // Format different data types
                if (typeof value === 'object' && value !== null) {
                    value = JSON.stringify(value);
                } else if (typeof value === 'string' && value.includes('@')) {
                    // Format email addresses
                    value = `<a href="mailto:${value}" style="color: #667eea;">${value}</a>`;
                } else if (key.toLowerCase().includes('date') || key.toLowerCase().includes('time')) {
                    // Format dates
                    value = this.formatDate(value);
                }
                
                html += `<td>${value}</td>`;
            });
            html += '</tr>';
        });
        html += '</tbody></table>';
        
        // Add pagination info if there are more records
        if (data.length > 10) {
            html += `<div style="text-align: center; margin-top: 15px; color: #666; font-style: italic;">
                        📄 Gösterilen: ${Math.min(10, data.length)} / ${data.length} kayıt
                     </div>`;
        }
        
        return html;
    }

    formatFieldName(fieldName) {
        // Convert field names to more readable format
        const fieldMappings = {
            'name': 'İsim',
            'surname': 'Soyisim',
            'age': 'Yaş',
            'email': 'E-posta',
            'city': 'Şehir',
            'created_at': 'Oluşturulma',
            'updated_at': 'Güncellenme'
        };
        
        return fieldMappings[fieldName] || fieldName.charAt(0).toUpperCase() + fieldName.slice(1);
    }

    formatDate(dateString) {
        try {
            const date = new Date(dateString);
            if (isNaN(date.getTime())) return dateString;
            
            return date.toLocaleString('tr-TR', {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch (e) {
            return dateString;
        }
    }

    showLoadingMessage() {
        this.updateStatus('processing');
        document.getElementById('loadingOverlay').style.display = 'flex';
        
        // Disable send button
        const sendBtn = document.getElementById('sendBtn');
        sendBtn.disabled = true;
        sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>İşleniyor...</span>';
    }

    hideLoadingMessage() {
        this.updateStatus('ready');
        document.getElementById('loadingOverlay').style.display = 'none';
        
        // Enable send button
        const sendBtn = document.getElementById('sendBtn');
        sendBtn.disabled = false;
        sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i><span>Gönder</span>';
    }

    async sendQuery() {
        if (this.isProcessing) return;

        const input = document.getElementById('queryInput');
        const query = input.value.trim();
        
        if (!query) {
            this.addMessage('❌ Lütfen bir sorgu yazın.');
            return;
        }
        
        this.isProcessing = true;
        
        // Add user message
        this.addMessage(query, true);
        input.value = '';
        
        // Hide suggestions
        document.getElementById('suggestions').style.display = 'none';
        
        // Show loading
        this.showLoadingMessage();
        
        try {
            const requestBody = { 
                query: query,
                selected_collection: this.selectedCollection 
            };
            
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            // Hide loading
            this.hideLoadingMessage();
            
            // Process result
            this.processQueryResult(result);
            
        } catch (error) {
            console.error('Query error:', error);
            this.hideLoadingMessage();
            
            const errorMessage = `❌ Bağlantı hatası: ${error.message}`;
            this.addMessage(errorMessage);
            this.updateStatus('error');
            
            // Reset status after 3 seconds
            setTimeout(() => this.updateStatus('ready'), 3000);
        } finally {
            this.isProcessing = false;
        }
    }

    processQueryResult(result) {
        let responseContent = '';
        
        if (result.error) {
            responseContent = `❌ Hata: ${result.error}`;
            this.updateStatus('error');
            setTimeout(() => this.updateStatus('ready'), 3000);
        } else {
            // Success case
            responseContent = result.response || 'Sorgu başarıyla tamamlandı';
            
            // Add data table if available
            if (result.data && Array.isArray(result.data) && result.data.length > 0) {
                responseContent += '<br><br>' + this.formatData(result.data);
            } else if (result.data && !Array.isArray(result.data)) {
                // Handle non-array data (like collection info)
                responseContent += '<br><br><pre>' + JSON.stringify(result.data, null, 2) + '</pre>';
            }
            
            this.updateStatus('ready');
        }
        
        this.addMessage(responseContent);
    }

    handleKeyPress(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendQuery();
        }
    }

    // Utility methods for future enhancements
    clearChat() {
        const chatContainer = document.getElementById('chatContainer');
        chatContainer.innerHTML = `
            <div class="message bot-message welcome-message">
                <div class="message-header">
                    <div class="avatar bot-avatar">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="message-info">
                        <span class="sender">MongoDB Agent</span>
                        <span class="timestamp">${new Date().toLocaleTimeString('tr-TR')}</span>
                    </div>
                </div>
                <div class="message-content">
                    <h4>🧹 Chat Temizlendi</h4>
                    <p>Yeni sorularınızı bekliyorum!</p>
                </div>
            </div>
        `;
        
        // Reset message count
        this.messageCount = 1;
        this.updateMessageCount();
        
        this.scrollToBottom();
    }

    exportChatHistory() {
        const messages = document.querySelectorAll('.message');
        let chatHistory = '';
        
        messages.forEach(message => {
            chatHistory += message.textContent + '\n\n';
        });
        
        const blob = new Blob([chatHistory], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `mongodb-chat-${new Date().toISOString().split('T')[0]}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    }
}

// Global functions for HTML event handlers
let mongoAgent;

function sendQuery() {
    mongoAgent.sendQuery();
}

function handleKeyPress(event) {
    mongoAgent.handleKeyPress(event);
}

function quickQuery(query) {
    document.getElementById('queryInput').value = query;
    mongoAgent.sendQuery();
}

function selectSuggestion(suggestion) {
    document.getElementById('queryInput').value = suggestion;
    document.getElementById('suggestions').style.display = 'none';
    document.getElementById('queryInput').focus();
}

function clearChat() {
    mongoAgent.clearChat();
}

function exportChat() {
    mongoAgent.exportChatHistory();
}

function toggleAutoScroll() {
    mongoAgent.autoScroll = !mongoAgent.autoScroll;
    const icon = document.getElementById('scrollIcon');
    if (mongoAgent.autoScroll) {
        icon.className = 'fas fa-arrow-down';
        mongoAgent.scrollToBottom();
    } else {
        icon.className = 'fas fa-pause';
    }
}

function onCollectionChange() {
    mongoAgent.onCollectionChange();
}

function refreshCollections() {
    mongoAgent.refreshCollections();
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    mongoAgent = new MongoDBAgent();
    console.log('✅ MongoDB Agent UI initialized successfully');
});

// Add some keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Ctrl+L to clear chat
    if (event.ctrlKey && event.key === 'l') {
        event.preventDefault();
        mongoAgent.clearChat();
    }
    
    // Ctrl+E to export chat
    if (event.ctrlKey && event.key === 'e') {
        event.preventDefault();
        mongoAgent.exportChatHistory();
    }
});

// Add service worker for offline capability (future enhancement)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        // navigator.serviceWorker.register('/sw.js');
    });
}
