// static/script.js

// Page Navigation
function showPage(pageId) {
    const pages = document.querySelectorAll('.page');
    pages.forEach(page => page.classList.remove('active'));
    document.getElementById(pageId + '-page').classList.add('active');
    return false;
}

// Login & Register Forms (giữ nguyên)
document.getElementById('login-form').addEventListener('submit', function(e) {
    e.preventDefault();
    showPage('dashboard');
});
document.getElementById('register-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const password = document.getElementById('register-password').value;
    const confirm = document.getElementById('register-confirm').value;
    if (password !== confirm) {
        alert('Mật khẩu không khớp!');
        return;
    }
    alert('Đăng ký thành công!');
    showPage('login');
});

// ================== IMAGE DETECTION ==================
const imageInput = document.getElementById('image-input');
const imageUploadArea = document.getElementById('image-upload-area');
const imagePlaceholder = document.getElementById('image-placeholder');
const imagePreview = document.getElementById('image-preview');
const previewImage = document.getElementById('preview-image');
const analyzeImageBtn = document.getElementById('analyze-image-btn');
const imageResult = document.getElementById('image-result');

imageUploadArea.addEventListener('click', () => {
    imageInput.click();
});

imageInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            imagePlaceholder.style.display = 'none';
            imagePreview.style.display = 'flex';
            analyzeImageBtn.style.display = 'block';
            imageResult.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
});

function changeImage() {
    imageInput.click();
}

async function analyzeImage() {
    analyzeImageBtn.disabled = true;
    analyzeImageBtn.textContent = 'Đang phân tích...';
    const file = imageInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("image", file);

    try {
        const res = await fetch("/analyze_image", { method: "POST", body: formData });
        const data = await res.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Hiển thị ảnh kết quả (có overlay)
        previewImage.src = data.image;
        imagePlaceholder.style.display = 'none';
        imagePreview.style.display = 'flex';

        // Hiển thị kết quả phân tích
        const result = {
            status: data.status === "ABNORMAL" ? "cheating" : "safe",
            confidence: Math.round(data.confidence * 100),
            details: data.details || []
        };
        displayImageResult(result);
    } catch (err) {
        alert("Lỗi kết nối đến server!");
    } finally {
        analyzeImageBtn.disabled = false;
        analyzeImageBtn.textContent = 'Phân tích hình ảnh';
    }
}

function displayImageResult(result) {
    const statusBadge = document.getElementById('image-status-badge');
    const confidence = document.getElementById('image-confidence');
    const progress = document.getElementById('image-progress');
    const details = document.getElementById('image-details');

    const statusTexts = { safe: 'An toàn', suspicious: 'Nghi ngờ', cheating: 'Gian lận' };
    const statusIcons = { safe: '✓', suspicious: '⚠', cheating: '✗' };

    statusBadge.className = 'status-badge status-' + result.status;
    statusBadge.innerHTML = statusIcons[result.status] + ' ' + statusTexts[result.status];

    confidence.textContent = result.confidence + '%';
    progress.style.width = result.confidence + '%';
    progress.className = 'progress-fill progress-' + result.status;

    details.innerHTML = '';
    result.details.forEach(detail => {
        const li = document.createElement('li');
        li.textContent = detail;
        details.appendChild(li);
    });

    imageResult.style.display = 'block';
}

// ================== VIDEO DETECTION ==================
const videoInput = document.getElementById('video-input');
const videoUploadArea = document.getElementById('video-upload-area');
const videoPlaceholder = document.getElementById('video-placeholder');
const videoPreview = document.getElementById('video-preview');
const previewVideo = document.getElementById('preview-video');
const analyzeVideoBtn = document.getElementById('analyze-video-btn');
const videoStats = document.getElementById('video-stats');
const videoTimeline = document.getElementById('video-timeline');
const timelineEvents = document.getElementById('timeline-events');
const analyzingIndicator = document.getElementById('analyzing-indicator');

videoUploadArea.addEventListener('click', () => {
    videoInput.click();
});

videoInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const url = URL.createObjectURL(file);
        previewVideo.src = url;
        videoPlaceholder.style.display = 'none';
        videoPreview.style.display = 'flex';
        analyzeVideoBtn.style.display = 'block';
        videoStats.style.display = 'none';
        videoTimeline.style.display = 'none';
        timelineEvents.innerHTML = '';
    }
});

function changeVideo() {
    videoInput.click();
}

async function analyzeVideo() {
    analyzeVideoBtn.style.display = 'none';
    videoTimeline.style.display = 'block';
    analyzingIndicator.style.display = 'block';
    timelineEvents.innerHTML = '';

    const file = videoInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("video", file);

    try {
        const res = await fetch("/analyze_video", { method: "POST", body: formData });
        const data = await res.json();

        // Hiển thị từng sự kiện
        data.events.forEach(addTimelineEvent);

        // Cập nhật thống kê
        document.getElementById('stat-total').textContent = data.stats.total;
        document.getElementById('stat-safe').textContent = data.stats.safe;
        document.getElementById('stat-suspicious').textContent = data.stats.suspicious;
        document.getElementById('stat-cheating').textContent = data.stats.cheating;
    } catch (err) {
        alert("Lỗi khi phân tích video. Hãy thử lại.");
    } finally {
        analyzingIndicator.style.display = 'none';
        videoStats.style.display = 'grid';
    }
}

function addTimelineEvent(event) {
    const statusTexts = { safe: 'An toàn', suspicious: 'Nghi ngờ', cheating: 'Gian lận' };
    const statusIcons = { safe: '✓', suspicious: '⚠', cheating: '✗' };

    const eventDiv = document.createElement('div');
    eventDiv.className = 'timeline-event';
    eventDiv.innerHTML = `
        <div class="timeline-event-badge">
            <span class="status-badge status-${event.status}">
                ${statusIcons[event.status]} ${statusTexts[event.status]}
            </span>
        </div>
        <div class="timeline-event-content">
            <div class="timeline-event-time">
                <svg class="icon-small" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="width: 0.75rem; height: 0.75rem;">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                ${event.time}
            </div>
            <p class="timeline-event-description">${event.description}</p>
        </div>
    `;
    timelineEvents.appendChild(eventDiv);
}