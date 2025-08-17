document.addEventListener('DOMContentLoaded', () => {
    // --- Get DOM Elements (Define them once at the top) ---
    const searchForm = document.getElementById('search-form');
    const toggleFiltersBtn = document.getElementById('toggle-filters-btn');
    const advancedFilters = document.getElementById('advanced-filters');
    const addObjectBtn = document.getElementById('add-object-btn');
    const objectList = document.getElementById('object-list');
    const objectSelect = document.getElementById('object-select');
    const objectCount = document.getElementById('object-count');
    const modalOverlay = document.getElementById('video-modal');
    const closeModalBtn = document.getElementById('close-modal-btn');
    const modalVideoPlayer = document.getElementById('modal-video-player');
    const modalVideoTitle = document.getElementById('modal-video-title');
    const resultsContainer = document.getElementById('results-container');
    
    // ====================================================================
    // EVENT LISTENERS
    // ====================================================================

    // --- Main Form Submission Listener (The single source of truth) ---
    searchForm.addEventListener('submit', (e) => {
        e.preventDefault(); // Prevent the browser from reloading the page
        
        // 1. Construct the query_data object from the current state of the form
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
        
        console.log('Sending to backend:', query_data);

        // 2. Call the function that handles the API request
        performSearch(query_data);
    });

    // --- UI Interaction Listeners ---

    // Toggle advanced filters visibility
    toggleFiltersBtn.addEventListener('click', () => {
        advancedFilters.classList.toggle('hidden');
        toggleFiltersBtn.textContent = advancedFilters.classList.contains('hidden') ? '▼ Advanced Filters' : '▲ Hide Filters';
    });

    // Add an object to the list
    addObjectBtn.addEventListener('click', () => {
        const label = objectSelect.value;
        const count = objectCount.value;

        // Prevent adding duplicates
        if (document.querySelector(`.object-item[data-label="${label}"]`)) {
            alert('Object already added.');
            return;
        }

        const objectItem = document.createElement('div');
        objectItem.classList.add('object-item');
        objectItem.setAttribute('data-label', label);
        objectItem.setAttribute('data-count', count);
        
        objectItem.innerHTML = `
            <span>${label} (Count: >= ${count})</span>
            <button type="button" class="remove-obj-btn">X</button>
        `;

        objectList.appendChild(objectItem);
    });

    // Remove an object from the list (using event delegation for efficiency)
    objectList.addEventListener('click', (e) => {
        if (e.target.classList.contains('remove-obj-btn')) {
            e.target.parentElement.remove();
        }
    });

    // ====================================================================
    // CORE FUNCTIONS
    // ====================================================================

    /**
     * Performs the actual search by sending data to the Flask backend API.
     * @param {object} query_data - The structured search query.
     */
    async function performSearch(query_data) {
        // Show a loading state to the user
        resultsContainer.innerHTML = '<p>Searching...</p>';
        
        try {
            // Use fetch() to send a POST request to our Flask server's /search endpoint
            const response = await fetch('/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(query_data) // Convert the JS object to a JSON string
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const results = await response.json();
            displayResults(results); // Call the function to render the results

        } catch (error) {
            console.error('Search failed:', error);
            resultsContainer.innerHTML = `<p style="color: red;">An error occurred: ${error}</p>`;
        }
    }

    // ====================================================================
    //              LOGIC CỦA MODAL VIDEO
    // ====================================================================

    // Mở modal khi một hình ảnh kết quả được nhấp (sử dụng ủy quyền sự kiện)
    resultsContainer.addEventListener('click', (e) => {
        // Kiểm tra xem phần tử được nhấp hoặc cha của nó có phải là hình ảnh kết quả không
        const resultImage = e.target.closest('.result-item-image');
        if (resultImage) {
            const videoId = resultImage.dataset.videoId;
            const keyframeIndex = parseInt(resultImage.dataset.keyframeIndex);
            
            // Giả định rằng bạn có một keyframe mỗi giây.
            // Nếu bạn có một keyframe mỗi 2 giây, hãy nhân với 2.
            const frameRate = 1; // 1 keyframe/giây
            let startTime = keyframeIndex * frameRate;

            // Bắt đầu video 5 giây trước khung hình chính để có ngữ cảnh
            startTime = Math.max(0, startTime - 5);

            openModal(videoId, startTime);
        }
    });

    // Đóng modal
    function closeModal() {
        modalOverlay.classList.add('hidden');
        // Dừng video và xóa nguồn để ngăn nó phát trong nền
        modalVideoPlayer.pause();
        modalVideoPlayer.src = ""; 
    }

    closeModalBtn.addEventListener('click', closeModal);

    // Đóng modal khi nhấp vào bên ngoài hộp nội dung
    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) {
            closeModal();
        }
    });

    // Mở modal và phát video
    function openModal(videoId, startTime) {
        modalVideoTitle.textContent = `Playing: ${videoId}`;
        
        // Xây dựng URL nguồn video, bao gồm thời gian bắt đầu
        // Media Fragments URI là một tiêu chuẩn web để liên kết đến các phần của phương tiện
        const videoUrl = `/videos/${videoId}#t=${startTime}`;
        
        modalVideoPlayer.src = videoUrl;
        
        modalOverlay.classList.remove('hidden');
        // Không cần gọi play() vì chúng ta có thuộc tính 'autoplay' trên thẻ video
    }

    /**
     * Renders the search results into the results container.
     * @param {Array} results - An array of result objects from the backend.
     */
    function displayResults(results) {
        if (!results || results.length === 0) {
            resultsContainer.innerHTML = '<p>No results found.</p>';
            return;
        }

        resultsContainer.innerHTML = '';

        results.forEach(item => {
            const resultElement = document.createElement('div');
            resultElement.classList.add('result-item');
            
            const imageUrl = `/frames/${item.video_id}/${item.keyframe_index}`;

            // CẬP NHẬT QUAN TRỌNG Ở ĐÂY:
            // Thêm một lớp và các thuộc tính dữ liệu vào thẻ img
            // để chúng ta có thể dễ dàng nhắm mục tiêu và lấy thông tin từ nó.
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
                    <p><strong>RRF Score: ${item.rrf_score.toFixed(4)}</strong></p>
                    <div class="result-scores">
                        Vector Dist: ${item.vector_score ? item.vector_score.toFixed(4) : 'N/A'}<br>
                        Content Score: ${item.content_score ? item.content_score.toFixed(2) : 'N/A'}<br>
                        Metadata Score: ${item.metadata_score ? item.metadata_score.toFixed(2) : 'N/A'}
                    </div>
                </div>
            `;
            resultsContainer.appendChild(resultElement);
        });
    }
});