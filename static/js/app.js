document.addEventListener('DOMContentLoaded', function() {
    // UI Elements
    const shapeFilter = document.getElementById('shape-filter');
    const colorCheckboxes = document.getElementById('color-checkboxes');
    const colorPalette = document.querySelector('.color-palette');
    const applyFiltersBtn = document.getElementById('apply-filters');
    const resetFiltersBtn = document.getElementById('reset-filters');
    const resultsContainer = document.getElementById('results-container');
    const resultCount = document.getElementById('result-count');
    const loadingIndicator = document.getElementById('loading-indicator');
    const gridViewBtn = document.getElementById('grid-view');
    const listViewBtn = document.getElementById('list-view');
    const themeSwitch = document.getElementById('theme-switch');
    const toggleFiltersBtn = document.getElementById('toggle-filters');
    const filterBody = document.querySelector('.filter-body');
    
    // Color mapping to hex values for visual indicators
    const colorMap = {
        'red': '#f44336',
        'green': '#4CAF50',
        'blue': '#2196F3',
        'yellow': '#FFEB3B',
        'black': '#000000',
        'white': '#ffffff',
        'orange': '#FF9800',
        'purple': '#9C27B0',
        'brown': '#795548',
        'pink': '#E91E63',
        'gray': '#9E9E9E',
        'grey': '#9E9E9E'
    };
    
    // Initialize the application
    function init() {
        loadOptions();
        setupEventListeners();
        loadThemePreference();
        
        // Show loading animation on page load
        showLoading(true);
        
        // Add animation classes
        document.body.classList.add('fade-in');
    }
    
    // Load available options from the API
    function loadOptions() {
        showLoading(true);
        fetch('/api/options')
            .then(response => response.json())
            .then(data => {
                populateShapeOptions(data.shapes);
                populateColorOptions(data.colors);
                showLoading(false);
                
                // Add animation to results container after options are loaded
                resultsContainer.classList.add('fade-in');
            })
            .catch(error => {
                console.error('Error loading options:', error);
                showLoading(false);
                showEmptyState('Failed to load options. Please refresh the page.');
            });
    }
    
    // Populate shape dropdown
    function populateShapeOptions(shapes) {
        shapeFilter.innerHTML = '<option value="">All Types</option>';
        shapes.forEach(shape => {
            const option = document.createElement('option');
            option.value = shape;
            option.textContent = capitalizeFirstLetter(shape);
            shapeFilter.appendChild(option);
        });
    }
    
    // Populate color options
    function populateColorOptions(colors) {
        // Clear existing options
        colorCheckboxes.innerHTML = '';
        colorPalette.innerHTML = '';
        
        // Sort colors for better organization
        colors.sort();
        
        // Add color palette options (visual color circles)
        colors.forEach(color => {
            const colorHex = colorMap[color.toLowerCase()] || '#cccccc';
            
            // Create color circle for the palette
            const colorOption = document.createElement('div');
            colorOption.className = 'color-option';
            colorOption.style.backgroundColor = colorHex;
            colorOption.dataset.color = color;
            colorOption.setAttribute('title', capitalizeFirstLetter(color));
            colorOption.addEventListener('click', function() {
                // Toggle the checkbox associated with this color
                const checkbox = document.querySelector(`#color-${color}`);
                if (checkbox) {
                    checkbox.checked = !checkbox.checked;
                    this.classList.toggle('selected', checkbox.checked);
                }
            });
            
            colorPalette.appendChild(colorOption);
            
            // Create checkbox for the color
            const colorCheckbox = document.createElement('div');
            colorCheckbox.className = 'color-checkbox';
            
            const input = document.createElement('input');
            input.type = 'checkbox';
            input.id = `color-${color}`;
            input.value = color;
            input.dataset.color = color;
            input.addEventListener('change', function() {
                // Update the color palette selection
                const colorCircle = document.querySelector(`.color-option[data-color="${color}"]`);
                if (colorCircle) {
                    colorCircle.classList.toggle('selected', this.checked);
                }
            });
            
            const label = document.createElement('label');
            label.htmlFor = `color-${color}`;
            
            const colorIndicator = document.createElement('span');
            colorIndicator.className = 'color-indicator';
            colorIndicator.style.backgroundColor = colorHex;
            
            const colorName = document.createElement('span');
            colorName.textContent = capitalizeFirstLetter(color);
            
            label.appendChild(colorIndicator);
            label.appendChild(colorName);
            
            colorCheckbox.appendChild(input);
            colorCheckbox.appendChild(label);
            
            colorCheckboxes.appendChild(colorCheckbox);
        });
    }
    
    // Set up event listeners
    function setupEventListeners() {
        // Apply filters button
        applyFiltersBtn.addEventListener('click', applyFilters);
        
        // Reset filters button
        resetFiltersBtn.addEventListener('click', resetFilters);
        
        // View toggle buttons
        gridViewBtn.addEventListener('click', () => {
            setViewMode('grid');
        });
        
        listViewBtn.addEventListener('click', () => {
            setViewMode('list');
        });
        
        // Theme toggle
        themeSwitch.addEventListener('click', toggleTheme);
        
        // Toggle filters visibility on mobile
        toggleFiltersBtn.addEventListener('click', toggleFilters);
        
        // Add keyboard navigation for accessibility
        document.addEventListener('keydown', handleKeyboardNavigation);
    }
    
    // Toggle between grid and list views
    function setViewMode(mode) {
        if (mode === 'grid') {
            resultsContainer.className = 'grid-view';
            gridViewBtn.classList.add('active');
            listViewBtn.classList.remove('active');
            localStorage.setItem('preferredView', 'grid');
        } else {
            resultsContainer.className = 'list-view';
            listViewBtn.classList.add('active');
            gridViewBtn.classList.remove('active');
            localStorage.setItem('preferredView', 'list');
        }
        
        // Add animation to results
        animateResults();
    }
    
    // Toggle theme between light and dark mode
    function toggleTheme() {
        const body = document.body;
        const isDarkMode = body.classList.contains('dark-mode');
        
        if (isDarkMode) {
            body.classList.remove('dark-mode');
            themeSwitch.innerHTML = '<i class="fas fa-moon"></i>';
            localStorage.setItem('theme', 'light');
        } else {
            body.classList.add('dark-mode');
            themeSwitch.innerHTML = '<i class="fas fa-sun"></i>';
            localStorage.setItem('theme', 'dark');
        }
    }
    
    // Load theme preference from localStorage
    function loadThemePreference() {
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            document.body.classList.add('dark-mode');
            themeSwitch.innerHTML = '<i class="fas fa-sun"></i>';
        } else {
            themeSwitch.innerHTML = '<i class="fas fa-moon"></i>';
        }
        
        // Load view preference
        const savedView = localStorage.getItem('preferredView');
        if (savedView === 'list') {
            setViewMode('list');
        }
    }
    
    // Toggle filters visibility (especially for mobile)
    function toggleFilters() {
        filterBody.classList.toggle('collapsed');
        
        const icon = toggleFiltersBtn.querySelector('i');
        if (filterBody.classList.contains('collapsed')) {
            icon.classList.remove('fa-chevron-down');
            icon.classList.add('fa-chevron-up');
        } else {
            icon.classList.remove('fa-chevron-up');
            icon.classList.add('fa-chevron-down');
        }
    }
    
    // Handle keyboard navigation
    function handleKeyboardNavigation(e) {
        // Pressing Enter when focused on a color option should toggle it
        if (e.key === 'Enter' && document.activeElement.classList.contains('color-option')) {
            document.activeElement.click();
        }
        
        // Pressing Enter when focused on apply filters button
        if (e.key === 'Enter' && document.activeElement === applyFiltersBtn) {
            applyFilters();
        }
    }
    
    // Apply the selected filters
    function applyFilters() {
        // Clear previous results immediately
        resultsContainer.innerHTML = '';
        resultCount.textContent = '0';
        
        // Force clear any cached images
        clearImageCache();
        
        // Reset server state first
        fetch('/api/reset', {
            method: 'POST',
            headers: {
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
        })
        .then(() => {
            // Continue with filter application after reset
            const selectedShape = shapeFilter.value;
            const selectedColors = Array.from(colorCheckboxes.querySelectorAll('input:checked'))
                .map(input => input.value);
            
            showLoading(true);
            
            // Determine which API endpoint to use based on selected filters
            let endpoint, requestBody;
            
            if (selectedShape && selectedColors.length > 0) {
                endpoint = '/api/filter/combined';
                requestBody = {
                    shape: selectedShape,
                    colors: selectedColors
                };
            } else if (selectedShape) {
                endpoint = '/api/filter/shape';
                requestBody = {
                    shape: selectedShape
                };
            } else if (selectedColors.length > 0) {
                endpoint = '/api/filter/color';
                requestBody = {
                    colors: selectedColors
                };
            } else {
                // Nothing selected, show message
                showEmptyState('Please select at least one filter');
                showLoading(false);
                return;
            }
            
            // Make API request with no-cache headers
            fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                },
                body: JSON.stringify(requestBody)
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showEmptyState(`Error: ${data.error}`);
                } else {
                    displayResults(data.results);
                    resultCount.textContent = data.count;
                }
                showLoading(false);
            })
            .catch(error => {
                console.error('Error applying filters:', error);
                showEmptyState('An error occurred while fetching results');
                showLoading(false);
            });
        })
        .catch(error => {
            console.error('Error resetting state before search:', error);
            showEmptyState('An error occurred while preparing search');
            showLoading(false);
        });
    }
    
    // Reset all filters
    function resetFilters() {
        // Clear UI state
        shapeFilter.value = '';
        
        const checkboxes = colorCheckboxes.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(checkbox => {
            checkbox.checked = false;
        });
        
        // Clear color palette selections
        const colorOptions = colorPalette.querySelectorAll('.color-option');
        colorOptions.forEach(option => {
            option.classList.remove('selected');
        });
        
        // Clear results display
        showEmptyState('Start exploring fashion items');
        resultCount.textContent = '0';
        
        // Force clear any cached images
        clearImageCache();
        
        // Call reset endpoint to clear server-side cache and temp files
        fetch('/api/reset', {
            method: 'POST',
            headers: {
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
        })
        .then(response => {
            if (!response.ok) {
                console.error('Error resetting server state');
            }
        })
        .catch(error => {
            console.error('Error calling reset endpoint:', error);
        });
    }
    
    // Clear image cache
    function clearImageCache() {
        // Find all images in the results container
        const images = resultsContainer.querySelectorAll('img');
        
        // Set their src to empty and remove from DOM to help browser garbage collect
        images.forEach(img => {
            img.src = '';
            img.remove();
        });
        
        // Clear the entire results container
        resultsContainer.innerHTML = '';
        
        // Try to force garbage collection by removing references
        if (window.performance && window.performance.memory) {
            console.log('Attempting to clear memory...');
            try {
                window.performance.memory.usedJSHeapSize;
            } catch (e) {
                console.error('Failed to access memory metrics', e);
            }
        }
    }
    
    // Display the filtered results with animation
    function displayResults(results) {
        // Clear previous results completely
        resultsContainer.innerHTML = '';
        
        if (!results || results.length === 0) {
            showEmptyState('No results found for the selected filters');
            return;
        }
        
        // Create and append result cards with staggered animation
        results.forEach((item, index) => {
            const card = document.createElement('div');
            card.className = 'item-card';
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            
            const imageContainer = document.createElement('div');
            imageContainer.className = 'item-image-container';
            
            const image = document.createElement('img');
            image.className = 'item-image';
            
            // Add cache-busting parameter to image URL
            const cacheBuster = Date.now();
            const imageUrl = item.image_url.includes('?') 
                ? `${item.image_url}&_cb=${cacheBuster}` 
                : `${item.image_url}?_cb=${cacheBuster}`;
            
            image.src = imageUrl;
            image.alt = item.class;
            image.loading = 'lazy';
            
            // Set onload and onerror handlers
            image.onload = () => {
                // Image loaded successfully
                setTimeout(() => {
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }, 50 * index); // Stagger the animations
            };
            
            image.onerror = () => {
                console.error(`Failed to load image: ${imageUrl}`);
                // Set a placeholder
                image.src = '/static/img/placeholder.svg';
                setTimeout(() => {
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }, 50 * index);
            };
            
            imageContainer.appendChild(image);
            
            const infoContainer = document.createElement('div');
            infoContainer.className = 'item-info';
            
            const itemType = document.createElement('div');
            itemType.className = 'item-type';
            itemType.textContent = capitalizeFirstLetter(item.class);
            
            const itemColors = document.createElement('div');
            itemColors.className = 'item-colors';
            
            item.colors.forEach(color => {
                const colorTag = document.createElement('span');
                colorTag.className = 'color-tag';
                
                // Add color indicator if available
                const colorHex = colorMap[color.toLowerCase()];
                if (colorHex) {
                    const colorDot = document.createElement('span');
                    colorDot.style.display = 'inline-block';
                    colorDot.style.width = '8px';
                    colorDot.style.height = '8px';
                    colorDot.style.borderRadius = '50%';
                    colorDot.style.backgroundColor = colorHex;
                    colorDot.style.marginRight = '4px';
                    colorTag.appendChild(colorDot);
                }
                
                const colorText = document.createElement('span');
                colorText.textContent = capitalizeFirstLetter(color);
                colorTag.appendChild(colorText);
                
                itemColors.appendChild(colorTag);
            });
            
            infoContainer.appendChild(itemType);
            infoContainer.appendChild(itemColors);
            
            // Add confidence score if available
            if (item.confidence !== null && item.confidence !== undefined) {
                const confidenceEl = document.createElement('div');
                confidenceEl.className = 'item-confidence';
                
                // Convert confidence to percentage
                const confidencePercent = Math.round(item.confidence * 100);
                
                const confidenceBar = document.createElement('div');
                confidenceBar.className = 'confidence-bar';
                
                const confidenceFill = document.createElement('div');
                confidenceFill.className = 'confidence-fill';
                confidenceFill.style.width = `${confidencePercent}%`;
                
                // Color based on confidence level
                if (confidencePercent >= 80) {
                    confidenceFill.style.backgroundColor = 'var(--success)';
                } else if (confidencePercent >= 60) {
                    confidenceFill.style.backgroundColor = 'var(--warning)';
                } else {
                    confidenceFill.style.backgroundColor = 'var(--danger)';
                }
                
                confidenceBar.appendChild(confidenceFill);
                
                const confidenceText = document.createElement('div');
                confidenceText.className = 'confidence-text';
                confidenceText.textContent = `Match: ${confidencePercent}%`;
                
                confidenceEl.appendChild(confidenceText);
                confidenceEl.appendChild(confidenceBar);
                infoContainer.appendChild(confidenceEl);
            }
            
            // Add metadata details if available
            if (item.metadata) {
                const metadataEl = document.createElement('div');
                metadataEl.className = 'item-metadata';
                
                // Create a quality badge
                if (item.metadata.quality_score) {
                    const qualityBadge = document.createElement('div');
                    qualityBadge.className = 'quality-badge';
                    
                    // Determine quality level
                    let qualityLevel = 'low';
                    if (item.metadata.quality_score >= 80) {
                        qualityLevel = 'high';
                    } else if (item.metadata.quality_score >= 60) {
                        qualityLevel = 'medium';
                    }
                    
                    qualityBadge.classList.add(`quality-${qualityLevel}`);
                    qualityBadge.innerHTML = `<i class="fas fa-star"></i> ${qualityLevel.charAt(0).toUpperCase() + qualityLevel.slice(1)} Match`;
                    
                    metadataEl.appendChild(qualityBadge);
                }
                
                // Add dimensions if available
                if (item.metadata.dimensions) {
                    const dimensionsEl = document.createElement('span');
                    dimensionsEl.className = 'metadata-dimensions';
                    dimensionsEl.innerHTML = `<i class="fas fa-ruler-combined"></i> ${item.metadata.dimensions}`;
                    metadataEl.appendChild(dimensionsEl);
                }
                
                infoContainer.appendChild(metadataEl);
            }
            
            card.appendChild(imageContainer);
            card.appendChild(infoContainer);
            
            // Add transition style
            card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            
            resultsContainer.appendChild(card);
        });
    }
    
    // Animate results with staggered effect
    function animateResults() {
        const cards = resultsContainer.querySelectorAll('.item-card');
        cards.forEach((card, index) => {
            // Reset animation
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            
            // Trigger reflow
            void card.offsetWidth;
            
            // Apply animation with delay
            setTimeout(() => {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 50 * index);
        });
    }
    
    // Show empty state with custom message
    function showEmptyState(message) {
        resultsContainer.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">
                    <i class="fas fa-search"></i>
                </div>
                <h3>${message}</h3>
                <p>Try adjusting your filter criteria</p>
            </div>
        `;
    }
    
    // Show/hide loading indicator
    function showLoading(isLoading) {
        if (isLoading) {
            loadingIndicator.classList.remove('hidden');
        } else {
            loadingIndicator.classList.add('hidden');
        }
    }
    
    // Helper function to capitalize first letter
    function capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1).toLowerCase();
    }
    
    // Initialize the application
    init();
}); 