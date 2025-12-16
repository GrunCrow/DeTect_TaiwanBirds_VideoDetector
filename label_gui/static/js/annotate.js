// ============================================================================
// ANNOTATION PAGE JAVASCRIPT
// ============================================================================

// State
const state = {
    frames: [],
    currentFrameIdx: 0,
    currentVideoIdx: 0,
    annotations: {},
    classes: [],
    zoom: 1.0,
    minZoom: 1.0,
    panX: 0,
    panY: 0,
    canvasWidth: 1920,
    canvasHeight: 1080,
    frameWidth: 1920,
    frameHeight: 1080
};

// Canvas & drawing
const canvas = document.getElementById('annotationCanvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let isPanning = false;
let drawStartX = 0;
let drawStartY = 0;
let panStartX = 0;
let panStartY = 0;
let tempRect = null;
let baseImage = null;

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    await loadClasses();
    await loadFirstVideo();
    setupEventListeners();
});

async function loadClasses() {
    try {
        const response = await fetch('/api/classes');
        const data = await response.json();
        state.classes = data.classes;
        
        const select = document.getElementById('classSelect');
        select.innerHTML = data.classes.map((cls, idx) => 
            `<option value="${idx}">${cls}</option>`
        ).join('');
    } catch (error) {
        console.error('Error loading classes:', error);
    }
}

async function loadFirstVideo() {
    showLoading('Loading video frames...');
    try {
        const response = await fetch('/api/video/0/frames');
        const data = await response.json();
        
        if (response.ok) {
            state.currentVideoIdx = 0;
            state.frames = data.frames;
            state.currentFrameIdx = 0;
            
            // Load detection image
            const detectionImg = document.getElementById('detectionImage');
            detectionImg.src = `/api/detection-image/${data.video_basename}`;
            
            // Update header
            document.getElementById('videoInfo').textContent = 
                `Video ${data.video_idx + 1}/${data.video_count}: ${data.video_basename}`;
            
            // Load first frame
            await loadFrame(0);
        } else {
            showError(data.error || 'Failed to load video');
        }
    } catch (error) {
        showError('Connection error: ' + error.message);
    } finally {
        hideLoading();
    }
}

// ============================================================================
// CANVAS DRAWING
// ============================================================================

async function loadFrame(frameIdx) {
    if (frameIdx < 0 || frameIdx >= state.frames.length) return;
    
    state.currentFrameIdx = frameIdx;
    const [frameNumber, frameName] = state.frames[frameIdx];
    
    showLoading('Loading frame...');
    
    try {
        // Load image
        const img = new Image();
        img.onload = async () => {
            baseImage = img;
            state.frameWidth = img.width;
            state.frameHeight = img.height;
            
            // Calculate canvas size to fit image
            const containerWidth = canvas.parentElement.clientWidth - 30;
            const containerHeight = canvas.parentElement.clientHeight - 30;
            
            state.canvasWidth = containerWidth;
            state.canvasHeight = containerHeight;
            
            canvas.width = state.canvasWidth;
            canvas.height = state.canvasHeight;
            
            // Calculate minimum zoom
            state.minZoom = Math.max(
                state.canvasWidth / state.frameWidth,
                state.canvasHeight / state.frameHeight
            );
            state.zoom = state.minZoom;
            state.panX = 0;
            state.panY = 0;
            
            // Load annotations
            const annoResponse = await fetch(`/api/frame/${frameName}/annotations`);
            const annoData = await annoResponse.json();
            state.annotations[frameName] = annoData.annotations || [];
            
            // Redraw
            redrawCanvas();
            updateFrameInfo();
            updateStats();
            
            hideLoading();
        };
        
        img.onerror = () => {
            showError('Failed to load image');
            hideLoading();
        };
        
        img.src = `/api/frame/${frameName}/image`;
    } catch (error) {
        showError('Error loading frame: ' + error.message);
        hideLoading();
    }
}

function redrawCanvas() {
    if (!baseImage) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, state.canvasWidth, state.canvasHeight);
    ctx.fillStyle = '#f3f4f6';
    ctx.fillRect(0, 0, state.canvasWidth, state.canvasHeight);
    
    // Draw image with zoom and pan
    ctx.save();
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.zoom, state.zoom);
    ctx.drawImage(baseImage, 0, 0);
    ctx.restore();
    
    // Draw annotations
    drawAnnotations();
}

function drawAnnotations() {
    const [frameNumber, frameName] = state.frames[state.currentFrameIdx];
    const annotations = state.annotations[frameName] || [];
    
    const colors = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#a855f7', '#ec4899'];
    
    annotations.forEach((anno, idx) => {
        const classId = anno.class_id;
        const color = colors[classId % colors.length];
        
        // Convert normalized coords to pixel coords
        const x = anno.x_center - anno.width / 2;
        const y = anno.y_center - anno.height / 2;
        
        // Convert to canvas coords with zoom
        const canvasX = x * state.frameWidth * state.zoom + state.panX;
        const canvasY = y * state.frameHeight * state.zoom + state.panY;
        const canvasW = anno.width * state.frameWidth * state.zoom;
        const canvasH = anno.height * state.frameHeight * state.zoom;
        
        // Draw rectangle
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(canvasX, canvasY, canvasW, canvasH);
        
        // Draw class label
        ctx.fillStyle = color;
        ctx.font = 'bold 14px Arial';
        ctx.fillText(anno.class_name, canvasX + 4, canvasY - 4);
    });
}

function updateFrameInfo() {
    const frameNum = state.currentFrameIdx + 1;
    const totalFrames = state.frames.length;
    document.getElementById('frameInfo').textContent = `Frame ${frameNum}/${totalFrames}`;
    document.getElementById('currentFrameDisplay').textContent = `${frameNum}/${totalFrames}`;
}

function updateStats() {
    document.getElementById('totalFrames').textContent = state.frames.length;
    let annotatedCount = 0;
    Object.values(state.annotations).forEach(annos => {
        if (annos.length > 0) annotatedCount++;
    });
    document.getElementById('annotatedCount').textContent = annotatedCount;
}

// ============================================================================
// ZOOM & PAN
// ============================================================================

function canvasToImageCoords(canvasX, canvasY) {
    const imgX = (canvasX - state.panX) / state.zoom;
    const imgY = (canvasY - state.panY) / state.zoom;
    return { imgX, imgY };
}

function resetZoom() {
    state.zoom = state.minZoom;
    state.panX = 0;
    state.panY = 0;
    redrawCanvas();
    document.getElementById('zoomLevel').textContent = `${Math.round(state.zoom * 100)}%`;
}

canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    
    const rect = canvas.getBoundingClientRect();
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;
    
    // Convert to image coords before zoom
    const { imgX, imgY } = canvasToImageCoords(canvasX, canvasY);
    
    // Zoom
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    let newZoom = state.zoom * zoomFactor;
    
    // Constrain zoom
    newZoom = Math.max(state.minZoom, Math.min(10, newZoom));
    
    // Adjust pan to keep mouse position stable
    const newCanvasX = imgX * newZoom + state.panX;
    const newCanvasY = imgY * newZoom + state.panY;
    state.panX += canvasX - newCanvasX;
    state.panY += canvasY - newCanvasY;
    
    state.zoom = newZoom;
    redrawCanvas();
    document.getElementById('zoomLevel').textContent = `${Math.round(state.zoom * 100)}%`;
}, { passive: false });

// ============================================================================
// DRAWING BOUNDING BOXES
// ============================================================================

canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;
    
    if (e.button === 2) {
        // Right click - pan
        isPanning = true;
        panStartX = canvasX;
        panStartY = canvasY;
    } else if (e.button === 0) {
        // Left click - draw
        isDrawing = true;
        drawStartX = canvasX;
        drawStartY = canvasY;
    }
});

canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;
    
    if (isPanning) {
        const dx = canvasX - panStartX;
        const dy = canvasY - panStartY;
        state.panX += dx;
        state.panY += dy;
        panStartX = canvasX;
        panStartY = canvasY;
        redrawCanvas();
    } else if (isDrawing) {
        redrawCanvas();
        
        // Draw preview rectangle
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.7)';
        ctx.lineWidth = 2;
        ctx.strokeRect(drawStartX, drawStartY, canvasX - drawStartX, canvasY - drawStartY);
    }
});

canvas.addEventListener('mouseup', (e) => {
    if (isPanning) {
        isPanning = false;
    } else if (isDrawing) {
        isDrawing = false;
        
        const rect = canvas.getBoundingClientRect();
        const endX = e.clientX - rect.left;
        const endY = e.clientY - rect.top;
        
        // Convert canvas coords to image coords
        const { imgX: x1, imgY: y1 } = canvasToImageCoords(drawStartX, drawStartY);
        const { imgX: x2, imgY: y2 } = canvasToImageCoords(endX, endY);
        
        // Normalize
        let minX = Math.min(x1, x2) / state.frameWidth;
        let minY = Math.min(y1, y2) / state.frameHeight;
        let maxX = Math.max(x1, x2) / state.frameWidth;
        let maxY = Math.max(y1, y2) / state.frameHeight;
        
        // Constrain to [0, 1]
        minX = Math.max(0, Math.min(1, minX));
        minY = Math.max(0, Math.min(1, minY));
        maxX = Math.max(0, Math.min(1, maxX));
        maxY = Math.max(0, Math.min(1, maxY));
        
        // Check minimum size
        const width = maxX - minX;
        const height = maxY - minY;
        
        if (width > 0.01 && height > 0.01) {
            const classId = parseInt(document.getElementById('classSelect').value);
            const annotation = {
                class_id: classId,
                class_name: state.classes[classId],
                x_center: (minX + maxX) / 2,
                y_center: (minY + maxY) / 2,
                width: width,
                height: height
            };
            
            const [frameNumber, frameName] = state.frames[state.currentFrameIdx];
            if (!state.annotations[frameName]) {
                state.annotations[frameName] = [];
            }
            state.annotations[frameName].push(annotation);
            
            redrawCanvas();
            updateStats();
        }
    }
});

canvas.addEventListener('contextmenu', (e) => e.preventDefault());

// ============================================================================
// KEYBOARD SHORTCUTS
// ============================================================================

document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') prevFrame();
    if (e.key === 'ArrowRight') nextFrame();
    if (e.key === 'Delete' || e.key === 'Backspace') deleteLastBox();
});

// ============================================================================
// NAVIGATION
// ============================================================================

async function prevFrame() {
    if (state.currentFrameIdx > 0) {
        await loadFrame(state.currentFrameIdx - 1);
    }
}

async function nextFrame() {
    if (state.currentFrameIdx < state.frames.length - 1) {
        await loadFrame(state.currentFrameIdx + 1);
    }
}

async function prevVideo() {
    if (state.currentVideoIdx > 0) {
        state.currentVideoIdx--;
        showLoading('Loading previous video...');
        try {
            const response = await fetch(`/api/video/${state.currentVideoIdx}/frames`);
            const data = await response.json();
            
            if (response.ok) {
                state.frames = data.frames;
                state.currentFrameIdx = 0;
                state.annotations = {};
                
                const detectionImg = document.getElementById('detectionImage');
                detectionImg.src = `/api/detection-image/${data.video_basename}`;
                
                document.getElementById('videoInfo').textContent = 
                    `Video ${data.video_idx + 1}/${data.video_count}: ${data.video_basename}`;
                
                await loadFrame(0);
            }
        } finally {
            hideLoading();
        }
    }
}

async function nextVideo() {
    showLoading('Loading next video...');
    try {
        const response = await fetch(`/api/video/${state.currentVideoIdx + 1}/frames`);
        const data = await response.json();
        
        if (response.ok) {
            state.currentVideoIdx++;
            state.frames = data.frames;
            state.currentFrameIdx = 0;
            state.annotations = {};
            
            const detectionImg = document.getElementById('detectionImage');
            detectionImg.src = `/api/detection-image/${data.video_basename}`;
            
            document.getElementById('videoInfo').textContent = 
                `Video ${data.video_idx + 1}/${data.video_count}: ${data.video_basename}`;
            
            await loadFrame(0);
        } else {
            showInfo('All videos have been processed!');
        }
    } catch (error) {
        if (error.message.includes('404')) {
            showInfo('All videos have been processed!');
        } else {
            showError('Error loading video: ' + error.message);
        }
    } finally {
        hideLoading();
    }
}

// ============================================================================
// ANNOTATION MANAGEMENT
// ============================================================================

function deleteLastBox() {
    const [frameNumber, frameName] = state.frames[state.currentFrameIdx];
    if (state.annotations[frameName] && state.annotations[frameName].length > 0) {
        state.annotations[frameName].pop();
        redrawCanvas();
        updateStats();
    }
}

function clearAllBoxes() {
    const [frameNumber, frameName] = state.frames[state.currentFrameIdx];
    state.annotations[frameName] = [];
    redrawCanvas();
    updateStats();
}

async function showFrameTime() {
    const [frameNumber, frameName] = state.frames[state.currentFrameIdx];
    try {
        const response = await fetch(`/api/video/${state.frames[state.currentFrameIdx][1].split('_')[0]}/timestamp/${frameNumber}`);
        // Note: This would need proper video basename extraction
        alert('Frame at: ' + frameNumber + ' (timing feature needs implementation)');
    } catch (error) {
        alert('Frame index: ' + frameNumber);
    }
}

// ============================================================================
// SAVE & EXIT
// ============================================================================

async function saveAndExit() {
    showConfirmation(
        'Save All Annotations?',
        'Are you sure? This will save all annotations and exit.',
        async () => {
            showLoading('Saving annotations...');
            try {
                const savePromises = [];
                
                for (const [frameName, annotations] of Object.entries(state.annotations)) {
                    const promise = fetch(`/api/frame/${frameName}/save-annotations`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ annotations })
                    });
                    savePromises.push(promise);
                }
                
                await Promise.all(savePromises);
                
                showSuccess('All annotations saved! Redirecting...');
                setTimeout(() => {
                    window.location.href = '/';
                }, 1500);
            } catch (error) {
                showError('Save error: ' + error.message);
            } finally {
                hideLoading();
            }
        }
    );
}

// ============================================================================
// UI HELPERS
// ============================================================================

function setupEventListeners() {
    // Canvas right-click support for panning
}

function showLoading(text) {
    document.getElementById('loadingText').textContent = text;
    document.getElementById('loadingModal').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loadingModal').classList.add('hidden');
}

function showError(message) {
    alert('Error: ' + message);
}

function showInfo(message) {
    alert(message);
}

function showSuccess(message) {
    alert('Success: ' + message);
}

function showConfirmation(title, message, onYes) {
    document.getElementById('confirmTitle').textContent = title;
    document.getElementById('confirmMessage').textContent = message;
    document.getElementById('confirmModal').classList.remove('hidden');
    
    window.confirmYes = onYes;
}

function confirmYes() {
    document.getElementById('confirmModal').classList.add('hidden');
    if (window.confirmYes) window.confirmYes();
}

function confirmNo() {
    document.getElementById('confirmModal').classList.add('hidden');
}
