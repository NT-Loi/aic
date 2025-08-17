document.addEventListener('DOMContentLoaded', () => {
    // ====================================================================
    // 1. GET ALL DOM ELEMENTS
    // ====================================================================
    const searchForm = document.getElementById('search-form');
    const toggleFiltersBtn = document.getElementById('toggle-filters-btn');
    const advancedFilters = document.getElementById('advanced-filters');
    const addObjectBtn = document.getElementById('add-object-btn');
    const objectList = document.getElementById('object-list');
    const objectSelect = document.getElementById('object-select');
    const objectCount = document.getElementById('object-count');
    const resultsContainer = document.getElementById('results-container');
    const sortBySelect = document.getElementById('sort-by-select');
    
    // Elements for the Video Modal
    const modalOverlay = document.getElementById('video-modal');
    const closeModalBtn = document.getElementById('close-modal-btn');
    const modalVideoPlayer = document.getElementById('modal-video-player');
    const modalVideoTitle = document.getElementById('modal-video-title');

    // ====================================================================
    // 2. STATE MANAGEMENT (Client-side cache for results)
    // ====================================================================
    let currentResults = [];

    // ====================================================================
    // 3. EVENT LISTENERS
    // ====================================================================

    /**
     * Main listener for the search form submission.
     */
    searchForm.addEventListener('submit', (e) => {
        e.preventDefault(); // Prevent the browser from reloading the page
        
        const formData = new FormData(searchForm);
        const query_data = {
            query: formData.get('query'),
            text: formData.get('text'),
            metadata: formData.get('metadata'),
            objects: []
        };
        
        document.querySelectorAll('.object-item').forEach(item => {
            query_data.objects.push([
                item.getAttribute('data-label'),
                parseInt(item.getAttribute('data-count'))
            ]);
        });
        
        console.log('Sending search request to backend:', query_data);
        performSearch(query_data);
    });

    /**
     * Listener for the sort dropdown to re-render results.
     */
    sortBySelect.addEventListener('change', () => {
        displayResults(currentResults);
    });

    /**
     * Listener to open the video modal when a result image is clicked.
     * Uses event delegation for efficiency.
     */
    resultsContainer.addEventListener('click', (e) => {
        const resultImage = e.target.closest('.result-item-image');
        if (resultImage) {
            const videoId = resultImage.dataset.videoId;
            const keyframeIndex = parseInt(resultImage.dataset.keyframeIndex);
            
            // Assuming 1 keyframe per second. Adjust if your rate is different.
            const frameRate = 1; 
            let startTime = keyframeIndex * frameRate;
            startTime = Math.max(0, startTime - 5); // Start 5s before for context

            openModal(videoId, startTime);
        }
    });

    /**
     * Listeners for UI interactions (filters, objects, closing modal).
     */
    toggleFiltersBtn.addEventListener('click', () => {
        advancedFilters.classList.toggle('hidden');
        toggleFiltersBtn.textContent = advancedFilters.classList.contains('hidden') ? '▼ Advanced Filters' : '▲ Hide Filters';
    });

    addObjectBtn.addEventListener('click', () => {
        const label = objectSelect.value;
        const count = objectCount.value;

        if (document.querySelector(`.object-item[data-label="${label}"]`)) {
            alert('Object already added.');
            return;
        }

        const objectItem = document.createElement('div');
        objectItem.classList.add('object-item');
        objectItem.setAttribute('data-label', label);
        objectItem.setAttribute('data-count', count);
        objectItem.innerHTML = `<span>${label} (Count: >= ${count})</span><button type="button" class="remove-obj-btn">X</button>`;
        objectList.appendChild(objectItem);
    });

    objectList.addEventListener('click', (e) => {
        if (e.target.classList.contains('remove-obj-btn')) {
            e.target.parentElement.remove();
        }
    });

    closeModalBtn.addEventListener('click', closeModal);
    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) {
            closeModal();
        }
    });

    // ====================================================================
    // 4. CORE FUNCTIONS
    // ====================================================================

    /**
     * Calls the backend API to perform a search and stores the results.
     * @param {object} query_data - The structured search query.
     */
    async function performSearch(query_data) {
        resultsContainer.innerHTML = '<p>Searching...</p>';
        
        try {
            const response = await fetch('/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(query_data)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const results = await response.json();
            currentResults = results; // Cache the results
            displayResults(currentResults); // Initial display

        } catch (error) {
            console.error('Search failed:', error);
            currentResults = []; // Clear cache on error
            resultsContainer.innerHTML = `<p style="color: red;">An error occurred: ${error}</p>`;
        }
    }

    /**
     * Sorts and renders the search results into the UI.
     * @param {Array} results - The array of result objects to display.
     */
    function displayResults(results) {
        if (!results || results.length === 0) {
            resultsContainer.innerHTML = '<p>No results found.</p>';
            return;
        }

        // --- Sorting Logic ---
        const sortBy = sortBySelect.value;
        const sortedResults = [...results]; // Create a copy to sort

        sortedResults.sort((a, b) => {
            // vector_score is a distance (lower is better)
            if (sortBy === 'vector_score') {
                const scoreA = a[sortBy] === null ? Infinity : a[sortBy];
                const scoreB = b[sortBy] === null ? Infinity : b[sortBy];
                return scoreA - scoreB; // Ascending sort
            } else {
                // All other scores are relevance scores (higher is better)
                const scoreA = a[sortBy] === null ? -Infinity : a[sortBy];
                const scoreB = b[sortBy] === null ? -Infinity : b[sortBy];
                return scoreB - scoreA; // Descending sort
            }
        });

        // --- Rendering Logic ---
        resultsContainer.innerHTML = ''; 

        sortedResults.forEach(item => {
            const resultElement = document.createElement('div');
            resultElement.classList.add('result-item');
            
            const imageUrl = `/frames/${item.video_id}/${item.keyframe_index}`;
            // const scoresHTML = `
            //     Rerank Score: ${item.rerank_score ? item.rerank_score.toFixed(4) : 'N/A'}<br>
            //     RRF Score: ${item.rrf_score ? item.rrf_score.toFixed(4) : 'N/A'}<br>
            //     Vector Dist: ${item.vector_score ? item.vector_score.toFixed(4) : 'N/A'}<br>
            //     Content Score: ${item.content_score ? item.content_score.toFixed(2) : 'N/A'}<br>
            //     Metadata Score: ${item.metadata_score ? item.metadata_score.toFixed(2) : 'N/A'}
            // `;

            resultElement.innerHTML = `
                <img 
                    src="${imageUrl}" 
                    alt="Frame from ${item.video_id}" 
                    class="result-item-image" 
                    data-video-id="${item.video_id}"
                    data-keyframe-index="${item.keyframe_index}"
                    onerror="this.onerror=null;this.src='/static/placeholder.png';"
                >
                <div class="result-info">
                    <h3>${item.video_id} / Frame ${item.keyframe_index}</h3>
                    <p class="${sortBy === 'rerank_score' ? 'sorted-by' : ''}">
                        <strong>Re-Rank Score: ${item.rerank_score ? item.rerank_score.toFixed(4) : 'N/A'}</strong>
                    </p>
                    <div class="result-scores">
                        <!-- This logic is a bit complex, but it highlights the right score -->
                        ${ ['rerank_score', 'rrf_score', 'vector_score', 'content_score', 'metadata_score'].map(score_name => {
                            const isSorted = sortBy === score_name;
                            const score_label_map = {
                                rerank_score: 'Rerank Score',
                                rrf_score: 'RRF Score',
                                vector_score: 'Vector Dist',
                                content_score: 'Content Score',
                                metadata_score: 'Metadata Score'
                            }
                            const label = score_label_map[score_name];
                            const value = item[score_name] ? item[score_name].toFixed(4) : 'N/A';
                            return `<span class="${isSorted ? 'sorted-by' : ''}">${label}: ${value}</span>`;
                        }).join('<br>')}
                    </div>
                </div>
            `;
            resultsContainer.appendChild(resultElement);
        });
    }

    /**
     * Opens the video player modal.
     * @param {string} videoId - The ID of the video to play.
     * @param {number} startTime - The time in seconds to start playback from.
     */
    function openModal(videoId, startTime) {
        modalVideoTitle.textContent = `Playing: ${videoId}`;
        const videoUrl = `/videos/${videoId}#t=${startTime}`;
        
        modalVideoPlayer.src = videoUrl;
        modalOverlay.classList.remove('hidden');
        modalVideoPlayer.play(); // Explicitly call play
    }

    /**
     * Closes the video player modal and stops playback.
     */
    function closeModal() {
        modalOverlay.classList.add('hidden');
        modalVideoPlayer.pause();
        modalVideoPlayer.src = ""; // Stop buffering
    }
});