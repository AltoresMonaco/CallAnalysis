// Modern Sidebar System v2.0
// Check if Sidebar already exists to prevent duplicate declaration
if (typeof Sidebar !== 'undefined') {
    console.log('[Sidebar] Sidebar already exists, checking if HTML exists...');
    // Check if the sidebar HTML actually exists in the DOM
    const existingSidebarHTML = document.getElementById('modernSidebar');
    if (!existingSidebarHTML) {
        console.log('[Sidebar] Sidebar object exists but HTML is missing, forcing initialization...');
        console.log('[Sidebar] About to call Sidebar.init()...');
        console.log('[Sidebar] Sidebar object:', Sidebar);
        console.log('[Sidebar] Sidebar.init function:', typeof Sidebar.init);
        
        Sidebar.isInitialized = false; // Reset the flag
        try {
            Sidebar.init();
            console.log('[Sidebar] Sidebar.init() call completed');
        } catch (error) {
            console.error('[Sidebar] Error calling Sidebar.init():', error);
        }
    } else {
        console.log('[Sidebar] Sidebar HTML exists, ensuring it\'s properly initialized...');
        Sidebar.updateSidebarState();
        Sidebar.highlightCurrentPage();
    }
} else {
    const Sidebar = {
        version: '2.0',
        isCollapsed: false,
        isInitialized: false,

        init() {
            console.log('=== SIDEBAR INIT FUNCTION STARTED ===');
            console.log('[Sidebar] This is the MODERN SIDEBAR v2.0 init function');
            console.log(`[Sidebar v${this.version}] Init called`);
            console.log('[Sidebar] Already initialized:', this.isInitialized);
            console.log('[Sidebar] Current URL:', window.location.href);
            console.log('[Sidebar] This object:', this);
            console.log('[Sidebar] This.init function:', typeof this.init);
            console.log('[Sidebar] Function name:', this.init.name);
            
            // Check if sidebar HTML already exists
            const existingSidebar = document.getElementById('modernSidebar');
            console.log('[Sidebar] Existing sidebar element:', existingSidebar);
            
            if (existingSidebar) {
                console.log('[Sidebar] Sidebar HTML already exists, updating state only');
                this.isInitialized = true;
                this.updateSidebarState();
                this.highlightCurrentPage();
                return;
            }

            console.log('[Sidebar] Sidebar HTML does not exist, creating new sidebar...');
            
            try {
                // Check if old sidebar elements exist and remove them
                console.log('[Sidebar] Step 1: Cleaning up old sidebar elements...');
                this.cleanupOldSidebar();
                
                // Create and inject sidebar HTML
                console.log('[Sidebar] Step 2: Creating sidebar HTML...');
                this.createSidebar();
                
                // Add event listeners
                console.log('[Sidebar] Step 3: Adding event listeners...');
                this.addEventListeners();
                
                // Set initial state
                console.log('[Sidebar] Step 4: Setting initial state...');
                this.isCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
                this.updateSidebarState();
                
                this.isInitialized = true;
                console.log('[Sidebar] Initialization complete');
                
                // Add a visual indicator that the new sidebar is loaded
                console.log('[Sidebar] Step 5: Adding version indicator...');
                this.addVersionIndicator();
                
                console.log('[Sidebar] All initialization steps completed successfully!');
            } catch (error) {
                console.error('[Sidebar] Error during initialization:', error);
                this.isInitialized = false;
            }
            
            console.log('=== SIDEBAR INIT FUNCTION ENDED ===');
        },

        addVersionIndicator() {
            // Add a small indicator to show the new sidebar is loaded
            const indicator = document.createElement('div');
            indicator.style.cssText = `
                position: fixed;
                top: 10px;
                right: 10px;
                background: #10b981;
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                z-index: 10000;
                pointer-events: none;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                font-weight: 600;
            `;
            indicator.textContent = `✅ Modern Sidebar v${this.version} Loaded`;
            document.body.appendChild(indicator);
            
            // Also add a console message
            console.log(`🎉 Modern Sidebar v${this.version} successfully loaded!`);
            console.log('📱 Features: Left-side, collapsible, modern design');
            console.log('⌨️  Shortcut: Ctrl+B to toggle sidebar');
            
            // Remove after 5 seconds
            setTimeout(() => {
                if (indicator.parentNode) {
                    indicator.remove();
                }
            }, 5000);
        },

        cleanupOldSidebar() {
            console.log('[Sidebar] Cleaning up old sidebar elements...');
            
            // Remove old sidebar elements
            const oldElements = [
                'sideMenu',
                'menuOverlay',
                'menu-toggle'
            ];
            
            oldElements.forEach(id => {
                const element = document.getElementById(id);
                if (element) {
                    console.log(`[Sidebar] Removing old element: ${id}`);
                    element.remove();
                }
            });
            
            // Remove old sidebar classes
            const oldSidebar = document.querySelector('.side-menu');
            if (oldSidebar) {
                console.log('[Sidebar] Removing old side-menu element');
                oldSidebar.remove();
            }
            
            // Remove old menu toggle
            const oldToggle = document.querySelector('.menu-toggle');
            if (oldToggle) {
                console.log('[Sidebar] Removing old menu-toggle element');
                oldToggle.remove();
            }
        },

        createSidebar() {
            console.log('[Sidebar] Creating new sidebar HTML...');
            
            // Check if sidebar already exists
            const existingSidebar = document.getElementById('modernSidebar');
            if (existingSidebar) {
                console.log('[Sidebar] Sidebar already exists, skipping creation');
                return;
            }
            
            console.log('[Sidebar] Building sidebar HTML...');
            const sidebarHTML = `
                <div id="modernSidebar" class="modern-sidebar ${this.isCollapsed ? 'collapsed' : ''}">
                    <div class="sidebar-header">
                        <div class="sidebar-logo">
                            <img src="/static/logo.png" alt="CCI" class="logo-img">
                            <span class="logo-text">CCI</span>
                        </div>
                        <button class="sidebar-toggle" id="sidebarToggle" title="Toggle Sidebar">
                            <i class="fas fa-chevron-left"></i>
                        </button>
                    </div>
                    
                    <nav class="sidebar-nav">
                        <div class="nav-section">
                            <h3 class="nav-section-title">Main</h3>
                            <a href="/" class="nav-item" data-page="dashboard">
                                <i class="fas fa-chart-line"></i>
                                <span class="nav-text">Dashboard</span>
                            </a>
                            <a href="/upload" class="nav-item" data-page="upload">
                                <i class="fas fa-upload"></i>
                                <span class="nav-text">Upload Files</span>
                            </a>
                            <a href="/results" class="nav-item" data-page="results">
                                <i class="fas fa-history"></i>
                                <span class="nav-text">Results</span>
                            </a>
                        </div>
                        
                        <div class="nav-section">
                            <h3 class="nav-section-title">Analysis</h3>
                            <a href="/agent" class="nav-item" data-page="agent">
                                <i class="fas fa-user-cog"></i>
                                <span class="nav-text">Agent Dashboard</span>
                            </a>
                            <a href="/manage" class="nav-item" data-page="manage">
                                <i class="fas fa-cogs"></i>
                                <span class="nav-text">Manage</span>
                            </a>
                        </div>
                        
                        <div class="nav-section">
                            <h3 class="nav-section-title">Tools</h3>
                            <a href="/browser" class="nav-item" data-page="browser">
                                <i class="fas fa-globe"></i>
                                <span class="nav-text">Browser</span>
                            </a>
                            <button class="nav-item settings-btn" id="settingsBtn">
                                <i class="fas fa-cog"></i>
                                <span class="nav-text">Settings</span>
                            </button>
                        </div>
                    </nav>
                    
                    <div class="sidebar-footer">
                        <div class="user-info">
                            <div class="user-avatar">
                                <i class="fas fa-user"></i>
                            </div>
                            <div class="user-details">
                                <span class="user-name">Admin User</span>
                                <span class="user-role">Administrator</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div id="sidebarOverlay" class="sidebar-overlay"></div>
            `;
            
            console.log('[Sidebar] HTML built, length:', sidebarHTML.length);
            console.log('[Sidebar] Inserting sidebar HTML into body...');
            console.log('[Sidebar] Body element:', document.body);
            
            document.body.insertAdjacentHTML('afterbegin', sidebarHTML);
            console.log('[Sidebar] HTML inserted into body');
            
            // Verify sidebar was created
            const sidebar = document.getElementById('modernSidebar');
            if (sidebar) {
                console.log('[Sidebar] Modern sidebar created successfully');
                console.log('[Sidebar] Sidebar element:', sidebar);
                console.log('[Sidebar] Sidebar computed styles:', window.getComputedStyle(sidebar));
                console.log('[Sidebar] Sidebar display:', window.getComputedStyle(sidebar).display);
                console.log('[Sidebar] Sidebar visibility:', window.getComputedStyle(sidebar).visibility);
                console.log('[Sidebar] Sidebar z-index:', window.getComputedStyle(sidebar).zIndex);
                console.log('[Sidebar] Sidebar position:', window.getComputedStyle(sidebar).position);
                
                // Add temporary debugging styles to make sure it's visible
                sidebar.style.border = '3px solid red';
                sidebar.style.backgroundColor = 'blue';
                console.log('[Sidebar] Added debugging styles - sidebar should be visible with red border and blue background');
            } else {
                console.error('[Sidebar] Failed to create modern sidebar - element not found after insertion');
                console.error('[Sidebar] All elements in body:', document.body.children);
            }
            
            // Add main content wrapper
            const mainContent = document.querySelector('.container') || document.querySelector('main') || document.body;
            if (mainContent && !mainContent.classList.contains('main-content')) {
                mainContent.classList.add('main-content');
                console.log('[Sidebar] Added main-content class');
            }
        },

        addEventListeners() {
            console.log('[Sidebar] Adding event listeners...');
            
            // Toggle button
            const toggleBtn = document.getElementById('sidebarToggle');
            if (toggleBtn) {
                console.log('[Sidebar] Toggle button found, adding click listener');
                toggleBtn.addEventListener('click', () => this.toggle());
            } else {
                console.error('[Sidebar] Toggle button not found!');
            }

            // Overlay click
            const overlay = document.getElementById('sidebarOverlay');
            if (overlay) {
                console.log('[Sidebar] Overlay found, adding click listener');
                overlay.addEventListener('click', () => this.collapse());
            } else {
                console.error('[Sidebar] Overlay not found!');
            }

            // Settings button
            const settingsBtn = document.getElementById('settingsBtn');
            if (settingsBtn) {
                console.log('[Sidebar] Settings button found, adding click listener');
                settingsBtn.addEventListener('click', () => this.openSettings());
            } else {
                console.error('[Sidebar] Settings button not found!');
            }

            // Keyboard shortcuts
            document.addEventListener('keydown', (e) => {
                if (e.ctrlKey && e.key === 'b') {
                    e.preventDefault();
                    this.toggle();
                }
            });

            // Resize handler
            window.addEventListener('resize', () => {
                if (window.innerWidth < 768) {
                    this.collapse();
                }
            });

            // Highlight current page
            this.highlightCurrentPage();
            
            // Add event listener for sidebar actions
            this.setupEventListeners();
            
            console.log('[Sidebar] Event listeners added successfully');
        },

        toggle() {
            this.isCollapsed = !this.isCollapsed;
            this.updateSidebarState();
            localStorage.setItem('sidebarCollapsed', this.isCollapsed);
        },

        expand() {
            this.isCollapsed = false;
            this.updateSidebarState();
            localStorage.setItem('sidebarCollapsed', false);
        },

        collapse() {
            this.isCollapsed = true;
            this.updateSidebarState();
            localStorage.setItem('sidebarCollapsed', true);
        },

        updateSidebarState() {
            const sidebar = document.getElementById('modernSidebar');
            const toggleBtn = document.getElementById('sidebarToggle');
            
            if (sidebar) {
                if (this.isCollapsed) {
                    sidebar.classList.add('collapsed');
                    document.body.classList.add('sidebar-collapsed');
                } else {
                    sidebar.classList.remove('collapsed');
                    document.body.classList.remove('sidebar-collapsed');
                }
            }
            
            if (toggleBtn) {
                const icon = toggleBtn.querySelector('i');
                if (icon) {
                    icon.className = this.isCollapsed ? 'fas fa-chevron-right' : 'fas fa-chevron-left';
                }
            }
        },

        highlightCurrentPage() {
            const currentPath = window.location.pathname;
            const navItems = document.querySelectorAll('.nav-item');
            
            navItems.forEach(item => {
                const href = item.getAttribute('href');
                if (href && href !== '#' && currentPath === href) {
                    item.classList.add('active');
                } else {
                    item.classList.remove('active');
                }
            });
        },

        openSettings() {
            // Dispatch custom event for settings
            const event = new CustomEvent('sidebarAction', {
                detail: { action: 'openSettings' }
            });
            document.dispatchEvent(event);
            
            // Handle settings based on current page
            const currentPage = window.location.pathname;
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
        },

        // Add event listener for sidebar actions
        setupEventListeners() {
            document.addEventListener('sidebarAction', (event) => {
                const { action } = event.detail;
                const currentPage = window.location.pathname;
                
                console.log(`[Sidebar] Received action: ${action} on page: ${currentPage}`);
                
                switch (action) {
                    case 'openSettings':
                        this.handleOpenSettings(currentPage);
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

        handleOpenSettings(currentPage) {
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

    // Legacy functions for backward compatibility
    function toggleMenu() {
        Sidebar.toggle();
    }

    function closeMenu() {
        Sidebar.collapse();
    }

    function sidebarOpenSettings() {
        Sidebar.openSettings();
    }

    function navigateToResults() {
        window.location.href = '/results';
    }

    function navigateToManage() {
        window.location.href = '/manage';
    }

    // Auto-initialize when DOM is ready
    document.addEventListener('DOMContentLoaded', function() {
        console.log('[Sidebar] DOM Content Loaded, initializing sidebar...');
        Sidebar.init();
    });

    // Also try to initialize immediately if DOM is already ready
    if (document.readyState === 'loading') {
        console.log('[Sidebar] DOM still loading, waiting for DOMContentLoaded...');
    } else {
        console.log('[Sidebar] DOM already ready, initializing immediately...');
        Sidebar.init();
    }

    // Force initialization function for debugging
    window.forceSidebarInit = function() {
        console.log('[Sidebar] Force initialization called');
        console.log('[Sidebar] Global Sidebar object:', window.Sidebar);
        if (window.Sidebar && typeof window.Sidebar.init === 'function') {
            window.Sidebar.init();
        } else {
            console.error('[Sidebar] Sidebar object or init function not available globally');
        }
    };

    // Test function to check sidebar state
    window.checkSidebarState = function() {
        console.log('[Sidebar] Checking sidebar state...');
        console.log('[Sidebar] Global Sidebar object:', window.Sidebar);
        console.log('[Sidebar] Sidebar HTML exists:', !!document.getElementById('modernSidebar'));
        console.log('[Sidebar] Sidebar isInitialized:', window.Sidebar ? window.Sidebar.isInitialized : 'N/A');
    };

    // Export for global access
    window.Sidebar = Sidebar;
} 