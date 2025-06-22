// Sidebar functionality
function toggleMenu() {
    const menu = document.getElementById('sideMenu');
    const overlay = document.getElementById('menuOverlay');
    const menuToggle = document.querySelector('.menu-toggle');
    
    if (menu.classList.contains('open')) {
        closeMenu();
    } else {
        menu.classList.add('open');
        overlay.classList.add('show');
        menuToggle.innerHTML = '<i class="fas fa-times"></i>';
    }
}

function closeMenu() {
    const menu = document.getElementById('sideMenu');
    const overlay = document.getElementById('menuOverlay');
    const menuToggle = document.querySelector('.menu-toggle');
    
    menu.classList.remove('open');
    overlay.classList.remove('show');
    menuToggle.innerHTML = '<i class="fas fa-bars"></i>';
}

// Close menu on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        closeMenu();
    }
});

// Event-driven communication for sidebar actions
function sidebarOpenSettings() {
    console.log('[Sidebar] Settings button clicked');
    // Dispatch a custom event that the main page can listen to
    const event = new CustomEvent('sidebarAction', {
        detail: { action: 'openSettings' }
    });
    document.dispatchEvent(event);
    console.log('[Sidebar] Settings event dispatched');
    closeMenu(); // Close menu after action
}

function navigateToResults() {
    const event = new CustomEvent('sidebarAction', {
        detail: { action: 'navigateToResults' }
    });
    document.dispatchEvent(event);
    closeMenu();
}

function navigateToManage() {
    const event = new CustomEvent('sidebarAction', {
        detail: { action: 'navigateToManage' }
    });
    document.dispatchEvent(event);
    closeMenu();
}

// Load sidebar when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    loadSidebar();
});

function loadSidebar() {
    fetch('/static/sidebar.html')
        .then(response => response.text())
        .then(html => {
            document.body.insertAdjacentHTML('afterbegin', html);
            console.log('[Sidebar] Loaded successfully');
        })
        .catch(error => {
            console.error('Error loading sidebar:', error);
        });
} 