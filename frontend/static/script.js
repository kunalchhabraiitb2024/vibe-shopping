document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    // Function to show typing indicator
    function showTypingIndicator() {
        const typingElement = document.createElement('div');
        typingElement.classList.add('typing-indicator');
        typingElement.id = 'typing-indicator';
        typingElement.innerHTML = `
            <span>Agent is thinking</span>
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        chatBox.appendChild(typingElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Function to hide typing indicator
    function hideTypingIndicator() {
        const typingElement = document.getElementById('typing-indicator');
        if (typingElement) {
            typingElement.remove();
        }
    }

    // Function to display a message
    function displayMessage(message, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message');
        messageElement.classList.add(sender + '-message');
        
        // Handle recommendation objects (list of items)
        if (Array.isArray(message) && message.length > 0 && typeof message[0] === 'object' && 'name' in message[0]) {
             const recContainer = document.createElement('div');
             recContainer.classList.add('recommendations');
             
             const title = document.createElement('h3');
             title.textContent = '🛍️ Perfect Matches Found!';
             title.style.margin = '0 0 15px 0';
             title.style.color = '#2d3748';
             recContainer.appendChild(title);
             
             message.forEach((rec, index) => {
                  const itemDiv = document.createElement('div');
                  itemDiv.classList.add('recommendation-item');
                  
                  const nameDiv = document.createElement('div');
                  nameDiv.classList.add('item-name');
                  nameDiv.textContent = rec.name;
                  
                  const priceDiv = document.createElement('div');
                  priceDiv.classList.add('item-price');
                  priceDiv.textContent = `$${rec.price}`;
                  
                  itemDiv.appendChild(nameDiv);
                  itemDiv.appendChild(priceDiv);
                  
                  // Add enhanced features for the first item (main recommendation)
                  if (index === 0) {
                      // Show outfit suggestions if available
                      if (rec.outfit_suggestions && rec.outfit_suggestions.length > 0) {
                          const outfitDiv = document.createElement('div');
                          outfitDiv.classList.add('outfit-suggestions');
                          outfitDiv.innerHTML = `
                              <div class="feature-title">✨ Complete the Look:</div>
                              <div class="suggestion-items">
                                  ${rec.outfit_suggestions.map(item => `<span class="suggestion-item">${item.name} ($${item.price})</span>`).join('')}
                              </div>
                          `;
                          itemDiv.appendChild(outfitDiv);
                      }
                      
                      // Show customers also bought if available
                      if (rec.customers_also_bought && rec.customers_also_bought.length > 0) {
                          const alsoDiv = document.createElement('div');
                          alsoDiv.classList.add('also-bought');
                          alsoDiv.innerHTML = `
                              <div class="feature-title">👥 Customers Also Bought:</div>
                              <div class="suggestion-items">
                                  ${rec.customers_also_bought.map(item => `<span class="suggestion-item">${item.name} ($${item.price})</span>`).join('')}
                              </div>
                          `;
                          itemDiv.appendChild(alsoDiv);
                      }
                      
                      // Show trending badge if applicable
                      if (rec.is_trending) {
                          const trendingDiv = document.createElement('div');
                          trendingDiv.classList.add('trending-badge');
                          trendingDiv.innerHTML = '🔥 Trending Now';
                          nameDiv.appendChild(trendingDiv);
                      }
                  }
                  
                  recContainer.appendChild(itemDiv);
             });
             
             messageElement.appendChild(recContainer);
        } 
        // Handle arrays of simple messages (follow-up questions)
        else if (Array.isArray(message)) {
             messageElement.innerHTML = message.join('<br>');
        } 
        else {
            messageElement.textContent = message;
        }
        
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Function to send message to backend
    async function sendMessageToBackend(query) {
        showTypingIndicator();
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query })
            });

            hideTypingIndicator();

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log("Backend response:", data);

            if (data.response_type === 'follow_up') {
                data.questions.forEach(question => {
                    displayMessage(question, 'agent');
                });
            } else if (data.response_type === 'recommendation') {
                if (data.justification) {
                    displayMessage(data.justification, 'agent');
                }
                if (data.recommendations && data.recommendations.length > 0) {
                    displayMessage(data.recommendations, 'agent');
                } else {
                    displayMessage("Sorry, I couldn't find any items matching your preferences. Try adjusting your requirements!", 'agent');
                }
            } else if (data.response_type === 'message') {
                // Handle simple message response type (e.g., for reset confirmation)
                displayMessage(data.message, 'agent');
            } else if (data.response_type === 'error') {
                displayMessage("Sorry, there was an error: " + data.message, 'agent');
            }

        } catch (error) {
            hideTypingIndicator();
            console.error('Error:', error);
            displayMessage('Sorry, there was an error communicating with the server. Please try again.', 'agent');
        }
    }

    // Function to handle sending message from input
    function handleSendMessage() {
        const query = userInput.value.trim();
        if (query) {
            displayMessage(query, 'user');
            userInput.value = '';
            sendMessageToBackend(query);
        }
    }

    // Event listeners
    sendButton.addEventListener('click', handleSendMessage);

    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            handleSendMessage();
        }
    });

    // Initial welcome message
    displayMessage('👋 Hi! I\'m your personal shopping vibe agent. Tell me what vibe you\'re going for today!\n\n💡 Try saying something like:\n• "Something cute for brunch"\n• "Professional but comfortable"\n• "Casual weekend vibes"\n• "Dressy for date night"\n\n💬 You can type "reset" anytime to start a new conversation!', 'agent');
}); 