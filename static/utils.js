// Shared utilities for the application
// This file contains functions that can be used across different pages

// Global event system for component communication
class EventBus {
    constructor() {
        this.events = {};
    }

    on(event, callback) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(callback);
    }

    emit(event, data) {
        if (this.events[event]) {
            this.events[event].forEach(callback => callback(data));
        }
    }

    off(event, callback) {
        if (this.events[event]) {
            this.events[event] = this.events[event].filter(cb => cb !== callback);
        }
    }
}

// Global event bus instance
window.appEventBus = new EventBus();

// Common UI utilities
const UI = {
    showStatus: function(message, type) {
        const indicator = document.getElementById('statusIndicator');
        if (indicator) {
            indicator.textContent = message;
            indicator.className = `status-indicator status-${type} show`;
            
            setTimeout(() => {
                indicator.classList.remove('show');
            }, 3000);
        }
    },

    showModal: function(modalId) {
        console.log(`[UI] Attempting to show modal: ${modalId}`);
        const modal = document.getElementById(modalId);
        if (modal) {
            console.log(`[UI] Modal element found, setting display to block`);
            modal.style.display = 'block';
            console.log(`[UI] Modal display style is now: ${modal.style.display}`);
        } else {
            console.error(`[UI] Modal element not found: ${modalId}`);
        }
    },

    hideModal: function(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.style.display = 'none';
        }
    },

    closeModalOnOutsideClick: function(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            window.addEventListener('click', function(event) {
                if (event.target === modal) {
                    UI.hideModal(modalId);
                }
            });
        }
    }
};

// Make UI utilities globally available
window.UI = UI;

// Common API utilities
const API = {
    async get(endpoint) {
        try {
            const response = await fetch(endpoint);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`API GET error for ${endpoint}:`, error);
            throw error;
        }
    },

    async post(endpoint, data) {
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`API POST error for ${endpoint}:`, error);
            throw error;
        }
    },

    async postFormData(endpoint, formData) {
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`API POST FormData error for ${endpoint}:`, error);
            throw error;
        }
    }
};

// Make API utilities globally available
window.API = API;

// Shared sidebar initialization
const Sidebar = {
    init: function() {
        this.setupEventListeners();
    },

    setupEventListeners: function() {
        document.addEventListener('sidebarAction', function(event) {
            const { action } = event.detail;
            const currentPage = window.location.pathname;
            
            console.log(`[Sidebar] Received action: ${action} on page: ${currentPage}`);
            
            switch (action) {
                case 'openSettings':
                    Sidebar.handleOpenSettings(currentPage);
                    break;
                case 'navigateToResults':
                    if (currentPage !== '/results') {
                        window.location.href = '/results';
                    }
                    break;
                case 'navigateToManage':
                    if (currentPage !== '/manage') {
                        window.location.href = '/manage';
                    }
                    break;
                default:
                    console.warn(`Unknown sidebar action: ${action}`);
            }
        });
    },

    handleOpenSettings: function(currentPage) {
        if (currentPage === '/') {
            // On home page, try to open settings modal directly
            console.log('[Sidebar] Opening settings modal on home page');
            if (typeof window.openSettings === 'function') {
                window.openSettings();
            } else {
                console.error('[Sidebar] openSettings function not available');
            }
        } else {
            // On other pages, redirect to home with settings hash
            console.log('[Sidebar] Redirecting to home page with settings hash');
            window.location.href = '/#settings';
        }
    }
};

// Make Sidebar utilities globally available
window.Sidebar = Sidebar; 