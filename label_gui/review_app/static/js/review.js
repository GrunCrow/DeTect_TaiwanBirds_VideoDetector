/**
 * Review Page JavaScript - Video review and relabeling functionality
 */

let currentVideo = null;
let currentFrameIndex = 0;
let videoFrames = [];
let classes = {};
let selectedBboxIndex = null;
let sessionChanges = [];
let currentAnnotator = null;
let videoChanges = {}; // Track changes per video: {videoName: [{frame, index, oldClass, newClass}, ...]}
let zoomLevel = 1.0; // Zoom level: 1.0 = 100%
const ZOOM_STEP = 0.25;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 4.0;

const classColors = {
    'Bat': '#ff0000',
    'Bird': '#00ff00',
    'Insect': '#0000ff',
    'Drone': '#ffff00',
    'Plane': '#ff00ff',
    'Other': '#00ffff',
    'Unknown': '#808080'
};

const ANNOTATOR_STORAGE_KEY = 'currentAnnotator';

// Function to darken a hex color
function darkenColor(hex, amount = 0.5) {
    // Remove the # if present
    hex = hex.replace('#', '');
    
    // Parse the hex color
    const r = Math.max(0, Math.floor(parseInt(hex.substring(0, 2), 16) * amount));
    const g = Math.max(0, Math.floor(parseInt(hex.substring(2, 4), 16) * amount));
    const b = Math.max(0, Math.floor(parseInt(hex.substring(4, 6), 16) * amount));
    
    // Convert back to hex
    return '#' + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
}

document.addEventListener('DOMContentLoaded', async () => {
    try {
        console.log('Starting Review app initialization...');
        
        // Load annotator from session storage
        currentAnnotator = sessionStorage.getItem(ANNOTATOR_STORAGE_KEY);
        const annotatorSelect = document.getElementById('annotator-select');
        console.log('Annotator select element:', annotatorSelect);
        
        if (annotatorSelect) {
            if (currentAnnotator) {
                annotatorSelect.value = currentAnnotator;
            }
            annotatorSelect.addEventListener('change', (e) => {
                currentAnnotator = e.target.value;
                console.log('Annotator changed to:', currentAnnotator);
                if (currentAnnotator) {
                    sessionStorage.setItem(ANNOTATOR_STORAGE_KEY, currentAnnotator);
                } else {
                    sessionStorage.removeItem(ANNOTATOR_STORAGE_KEY);
                }
            });
        } else {
            console.warn('Annotator select element not found');
        }

        // Load classes
        console.log('Fetching classes...');
        const classesResponse = await fetch('/api/classes');
        if (!classesResponse.ok) {
            throw new Error(`Classes API returned ${classesResponse.status}`);
        }
        classes = await classesResponse.json();
        console.log('Classes loaded:', classes);
        populateClassSelect();

        // Load videos
        console.log('Fetching videos...');
        const videosResponse = await fetch('/api/videos');
        if (!videosResponse.ok) {
            throw new Error(`Videos API returned ${videosResponse.status}`);
        }
        const videos = await videosResponse.json();
        console.log('Videos loaded:', videos.length);
        displayVideos(videos.filter(v => v.annotated_frames > 0)); // Show only videos with annotations

        // Event listeners
        const showBboxesCheckbox = document.getElementById('show-bboxes-checkbox');
        const prevFrameBtn = document.getElementById('prev-frame-btn');
        const nextFrameBtn = document.getElementById('next-frame-btn');
        const confirmChangeBtn = document.getElementById('confirm-change-btn');
        const cancelChangeBtn = document.getElementById('cancel-change-btn');
        const clearBtn = document.getElementById('clear-btn');
        const zoomInBtn = document.getElementById('zoom-in-btn');
        const zoomOutBtn = document.getElementById('zoom-out-btn');
        const zoomResetBtn = document.getElementById('zoom-reset-btn');

        if (showBboxesCheckbox) showBboxesCheckbox.addEventListener('change', redrawFrame);
        if (prevFrameBtn) prevFrameBtn.addEventListener('click', previousFrame);
        if (nextFrameBtn) nextFrameBtn.addEventListener('click', nextFrame);
        if (confirmChangeBtn) confirmChangeBtn.addEventListener('click', confirmClassChange);
        if (cancelChangeBtn) cancelChangeBtn.addEventListener('click', cancelClassChange);
        if (clearBtn) clearBtn.addEventListener('click', clearChanges);
        if (zoomInBtn) zoomInBtn.addEventListener('click', zoomIn);
        if (zoomOutBtn) zoomOutBtn.addEventListener('click', zoomOut);
        if (zoomResetBtn) zoomResetBtn.addEventListener('click', zoomReset);

        // Setup modal listeners
        const closeHistoryBtn = document.getElementById('close-history-modal');
        const historyModal = document.getElementById('review-history-modal');
        if (closeHistoryBtn) {
            closeHistoryBtn.addEventListener('click', () => {
                historyModal.style.display = 'none';
            });
        }
        if (historyModal) {
            historyModal.addEventListener('click', (e) => {
                if (e.target === historyModal) {
                    historyModal.style.display = 'none';
                }
            });
        }

        console.log('✓ Review app initialized successfully');
    } catch (error) {
        console.error('Error initializing review:', error);
        const videoList = document.getElementById('video-list');
        if (videoList) {
            videoList.innerHTML = `<p style="color: red;">Error loading app: ${error.message}</p>`;
        }
        alert('Error loading app: ' + error.message);
    }
});

function populateClassSelect() {
    const select = document.getElementById('new-class-select');
    for (const [id, name] of Object.entries(classes)) {
        const option = document.createElement('option');
        option.value = id;
        option.textContent = name;
        select.appendChild(option);
    }
}

async function displayVideos(videos) {
    const videoList = document.getElementById('video-list');
    videoList.innerHTML = '';
    
    // Store videos for later reference
    currentVideos = videos;
    
    console.log('Displaying videos:', videos);
    console.log('Videos with relabel_history:', videos.filter(v => v.relabel_history));

    videos.forEach(video => {
        const videoItem = document.createElement('div');
        videoItem.className = 'video-item';
        videoItem.dataset.videoName = video.name;
        if (video.flagged) {
            videoItem.classList.add('flagged-video');
        }
        const changesInVideo = videoChanges[video.name]?.length || 0;
        
        let reviewBadgeHTML = '';
        if (video.relabel_history) {
            const reviewers = video.relabel_history.reviewers.join(', ');
            const changeCount = video.relabel_history.changes.length;
            reviewBadgeHTML = `<span class="review-badge" title="Reviewed by ${reviewers} (${changeCount} changes)">✅ ${reviewers}</span>`;
            console.log(`Adding review badge for ${video.name}: ${reviewers}`);
        }
        
        let flagHTML = '';
        if (video.flagged) {
            const annotator = video.flag_info?.annotator || 'Unknown';
            flagHTML = `<span class="flag-badge" title="Flagged by ${annotator} for bbox review">🚩 Needs BB Review</span>`;
        }
        
        videoItem.innerHTML = `
            <div class="video-header">
                <div class="video-name">
                    ${video.name}
                    ${changesInVideo > 0 ? ` <span class="change-badge">${changesInVideo}</span>` : ''}
                </div>
                <label class="flag-checkbox" title="Flag this video for bbox review">
                    <input type="checkbox" ${video.flagged ? 'checked' : ''} data-video-name="${video.name}">
                    <span>🚩</span>
                </label>
            </div>
            ${reviewBadgeHTML}
            ${flagHTML}
            <div class="video-stats">
                <span title="Frames with targets">🎯 ${video.annotated_frames}</span>
                <span title="Background frames">🌫️ ${video.background_frames}</span>
            </div>
        `;
        
        // Add event listener for flag checkbox
        const flagCheckbox = videoItem.querySelector('.flag-checkbox input');
        if (flagCheckbox) {
            flagCheckbox.addEventListener('change', (e) => {
                e.stopPropagation();
                toggleVideoFlag(video.name, e.target.checked, e.target);
            });
            // Prevent checkbox click from triggering video selection
            flagCheckbox.parentElement.addEventListener('click', (e) => {
                e.stopPropagation();
            });
        }
        
        videoItem.addEventListener('click', () => selectVideo(video.name));
        videoList.appendChild(videoItem);
    });
    
    // Attach event listeners to review badges
    attachHistoryListeners();
}

async function selectVideo(videoName) {
    currentVideo = videoName;
    currentFrameIndex = 0;
    selectedBboxIndex = null;
    document.getElementById('class-change-panel').style.display = 'none';

    // Update UI - highlight the active video
    document.querySelectorAll('.video-item').forEach(item => {
        item.classList.remove('active');
    });
    
    const activeVideoItem = document.querySelector(`.video-item[data-video-name="${videoName}"]`);
    if (activeVideoItem) {
        activeVideoItem.classList.add('active');
        // Scroll the active video into view
        activeVideoItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Load frames for this video
    try {
        const response = await fetch(`/api/video/${videoName}`);
        videoFrames = await response.json();

        // Filter to only frames with annotations
        videoFrames = videoFrames.filter(f => f.has_annotations);

        if (videoFrames.length === 0) {
            alert('No annotated frames in this video');
            return;
        }

        // Initialize video changes tracking if not exists
        if (!videoChanges[videoName]) {
            videoChanges[videoName] = [];
        }

        displayFrame(0);
    } catch (error) {
        console.error('Error loading video frames:', error);
        alert('Error loading frames: ' + error.message);
    }
}

async function displayFrame(frameIndex) {
    if (!videoFrames || frameIndex < 0 || frameIndex >= videoFrames.length) {
        console.log('Invalid frame index:', frameIndex, 'total frames:', videoFrames.length);
        return;
    }

    currentFrameIndex = frameIndex;
    const frame = videoFrames[frameIndex];

    console.log('Displaying frame:', frameIndex, frame);

    // Update frame info
    document.getElementById('frame-info').textContent =
        `${currentVideo} - Frame ${frame.frame_number} (${frameIndex + 1}/${videoFrames.length})`;
    document.getElementById('frame-counter').textContent = `${frameIndex + 1} / ${videoFrames.length}`;

    // Update button states
    document.getElementById('prev-frame-btn').disabled = frameIndex === 0;
    document.getElementById('next-frame-btn').disabled = frameIndex === videoFrames.length - 1;

    // Load and display image
    const showBboxes = document.getElementById('show-bboxes-checkbox').checked;
    const imageUrl = `/api/frame/image/${encodeURIComponent(frame.image_path)}?draw_bboxes=false`;
    
    console.log('Loading image from:', imageUrl);

    const img = new Image();
    img.onerror = (error) => {
        console.error('Image load error:', error, 'URL:', imageUrl);
        document.getElementById('no-frame-placeholder').textContent = `❌ Failed to load image: ${frame.image_path}`;
    };
    img.onload = () => {
        console.log('Image loaded successfully, size:', img.width, 'x', img.height);
        const canvas = document.getElementById('frame-canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        // Draw bboxes with click detection if enabled
        if (showBboxes && frame.bboxes.length > 0) {
            console.log('Drawing', frame.bboxes.length, 'bboxes');
            drawBboxesWithInteraction(canvas, frame.bboxes, frame.image_path);
        }

        // Hide placeholder
        document.getElementById('no-frame-placeholder').style.display = 'none';
        canvas.style.display = 'block';
    };
    img.src = imageUrl;

    // Display bboxes list
    displayBboxesList(frame);
}

function drawBboxesWithInteraction(canvas, bboxes, imagePath) {
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    bboxes.forEach((bbox, idx) => {
        const x1 = (bbox.x_center - bbox.width / 2) * canvas.width;
        const y1 = (bbox.y_center - bbox.height / 2) * canvas.height;
        const x2 = (bbox.x_center + bbox.width / 2) * canvas.width;
        const y2 = (bbox.y_center + bbox.height / 2) * canvas.height;

        const isSelected = idx === selectedBboxIndex;
        let color = classColors[bbox.class_name] || '#00ff00';
        
        // Darken color if selected
        if (isSelected) {
            color = darkenColor(color, 0.3); // Darken by 70% (30% brightness remaining)
        }
        
        const lineWidth = isSelected ? 4 : 2;

        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Draw label
        const label = bbox.class_name;
        ctx.font = 'bold 14px Arial';
        ctx.fillStyle = color;
        ctx.fillRect(x1, Math.max(0, y1 - 20), 120, 20);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, x1 + 4, Math.max(15, y1 - 3));
    });

    // Add click handler (only once per canvas by removing old listeners)
    canvas.onclick = (e) => {
        const rect = canvas.getBoundingClientRect();
        const clickX = (e.clientX - rect.left) * scaleX;
        const clickY = (e.clientY - rect.top) * scaleY;

        // Check which bbox was clicked
        for (let i = 0; i < bboxes.length; i++) {
            const bbox = bboxes[i];
            const x1 = (bbox.x_center - bbox.width / 2) * canvas.width;
            const y1 = (bbox.y_center - bbox.height / 2) * canvas.height;
            const x2 = (bbox.x_center + bbox.width / 2) * canvas.width;
            const y2 = (bbox.y_center + bbox.height / 2) * canvas.height;

            if (clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2) {
                selectBbox(i, bbox, imagePath);
                return;
            }
        }
    };
}

function displayBboxesList(frame) {
    const bboxList = document.getElementById('bboxes-list');
    bboxList.innerHTML = '';

    if (frame.bboxes.length === 0) {
        bboxList.innerHTML = '<p class="placeholder-text">No targets in this frame</p>';
        return;
    }

    frame.bboxes.forEach((bbox, idx) => {
        const item = document.createElement('div');
        item.className = 'bbox-item ' + (idx === selectedBboxIndex ? 'selected' : '');
        item.innerHTML = `
            <div class="bbox-color" style="background-color: ${classColors[bbox.class_name] || '#00ff00'}"></div>
            <div class="bbox-info">
                <div class="bbox-id">BBox #${idx + 1}</div>
                <div class="bbox-class">${bbox.class_name}</div>
            </div>
        `;
        item.addEventListener('click', () => {
            const imagePath = videoFrames[currentFrameIndex].image_path;
            selectBbox(idx, bbox, imagePath);
        });
        bboxList.appendChild(item);
    });
}

function selectBbox(index, bbox, imagePath) {
    selectedBboxIndex = index;
    const frame = videoFrames[currentFrameIndex];

    // Update UI
    document.querySelectorAll('.bbox-item').forEach((item, i) => {
        item.classList.toggle('selected', i === index);
    });

    // Show change panel
    document.getElementById('class-change-panel').style.display = 'block';
    document.getElementById('current-class').textContent = bbox.class_name;
    document.getElementById('new-class-select').value = bbox.class_id;

    // Redraw frame to highlight selected bbox
    redrawFrame();
}

async function confirmClassChange() {
    if (!currentAnnotator) {
        alert('⚠️ Please select an annotator name first');
        document.getElementById('annotator-select').focus();
        return;
    }

    if (selectedBboxIndex === null) {
        alert('No bbox selected');
        return;
    }

    const frame = videoFrames[currentFrameIndex];
    const bbox = frame.bboxes[selectedBboxIndex];
    const newClassId = parseInt(document.getElementById('new-class-select').value);
    const applyToAll = document.getElementById('apply-to-all-frames-checkbox').checked;

    if (!newClassId && newClassId !== 0) {
        alert('Please select a new class');
        return;
    }

    const newClassName = classes[newClassId];
    const oldClassName = bbox.class_name;

    if (oldClassName === newClassName) {
        alert('No change: new class is the same as current class');
        return;
    }

    try {
        // Update the bbox
        const response = await fetch('/api/frame/update-class', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'X-Annotator': currentAnnotator || 'Unknown'
            },
            body: JSON.stringify({
                frame_image_path: frame.image_path,
                bbox_index: selectedBboxIndex,
                new_class_id: newClassId,
                old_class_name: oldClassName
            })
        });

        const result = await response.json();

        if (response.ok) {
            // Update local frame data
            bbox.class_id = newClassId;
            bbox.class_name = newClassName;

            // Track change
            trackChange(currentVideo, frame.stem, frame.image_path, selectedBboxIndex, oldClassName, newClassName);

            if (applyToAll) {
                // Apply to all frames with the same class
                const framesUpdated = await applyChangeToAllFrames(oldClassName, newClassId, newClassName);
                alert(`✅ Changed BBox #${selectedBboxIndex + 1} to ${newClassName} and ${framesUpdated - 1} other occurrences in this video`);
            } else {
                alert(`✅ Changed BBox #${selectedBboxIndex + 1} to ${newClassName}`);
            }

            // Update UI
            updateChangesCount();
            displayBboxesList(frame);

            // Cancel selection
            cancelClassChange();

            // Redraw
            redrawFrame();
            
            // Refresh the video list to show updated badges
            await refreshVideosList();
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        console.error('Error updating class:', error);
        alert('Error: ' + error.message);
    }
}

function trackChange(video, frameStem, imagePath, bboxIndex, oldClass, newClass) {
    // Add to session changes
    sessionChanges.push({
        video: video,
        frame: frameStem,
        image_path: imagePath,
        bbox_index: bboxIndex,
        old_class: oldClass,
        new_class: newClass,
        annotator: currentAnnotator
    });

    // Track in video changes
    if (!videoChanges[video]) {
        videoChanges[video] = [];
    }
    videoChanges[video].push({
        frame: frameStem,
        image_path: imagePath,
        bbox_index: bboxIndex,
        oldClass: oldClass,
        newClass: newClass
    });
}

async function applyChangeToAllFrames(oldClassName, newClassId, newClassName) {
    let framesUpdated = 0;

    for (let i = 0; i < videoFrames.length; i++) {
        const frame = videoFrames[i];
        for (let j = 0; j < frame.bboxes.length; j++) {
            const bbox = frame.bboxes[j];
            if (bbox.class_name === oldClassName) {
                try {
                    const response = await fetch('/api/frame/update-class', {
                        method: 'POST',
                        headers: { 
                            'Content-Type': 'application/json',
                            'X-Annotator': currentAnnotator || 'Unknown'
                        },
                        body: JSON.stringify({
                            frame_image_path: frame.image_path,
                            bbox_index: j,
                            new_class_id: newClassId,
                            old_class_name: oldClassName
                        })
                    });

                    if (response.ok) {
                        bbox.class_id = newClassId;
                        bbox.class_name = newClassName;
                        trackChange(currentVideo, frame.stem, frame.image_path, j, oldClassName, newClassName);
                        framesUpdated++;
                    }
                } catch (error) {
                    console.error(`Error updating frame ${i} bbox ${j}:`, error);
                }
            }
        }
    }

    return framesUpdated;
}

function cancelClassChange() {
    selectedBboxIndex = null;
    document.getElementById('apply-to-all-frames-checkbox').checked = false;
    document.getElementById('class-change-panel').style.display = 'none';
    displayBboxesList(videoFrames[currentFrameIndex]);
}

function previousFrame() {
    if (currentFrameIndex > 0) {
        selectedBboxIndex = null;
        document.getElementById('class-change-panel').style.display = 'none';
        displayFrame(currentFrameIndex - 1);
    }
}

function nextFrame() {
    if (currentFrameIndex < videoFrames.length - 1) {
        selectedBboxIndex = null;
        document.getElementById('class-change-panel').style.display = 'none';
        displayFrame(currentFrameIndex + 1);
    }
}

function redrawFrame() {
    if (currentFrameIndex < videoFrames.length) {
        displayFrame(currentFrameIndex);
    }
}

function updateChangesCount() {
    document.getElementById('changes-count').textContent = sessionChanges.length;
    
    // Count videos with changes
    let videosWithChanges = 0;
    for (const video in videoChanges) {
        if (videoChanges[video].length > 0) {
            videosWithChanges++;
        }
    }
    document.getElementById('video-changes-count').textContent = videosWithChanges;
}

async function exportChanges() {
    if (!currentAnnotator) {
        alert('⚠️ Please select an annotator name first');
        document.getElementById('annotator-select').focus();
        return;
    }

    if (sessionChanges.length === 0) {
        alert('No changes to export');
        return;
    }

    try {
        const response = await fetch('/api/changes/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                changes: sessionChanges,
                annotator: currentAnnotator
            })
        });

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `label_changes_${currentAnnotator}_${new Date().toISOString().slice(0, 10)}.csv`;
        a.click();
        window.URL.revokeObjectURL(url);
        alert('✅ Changes exported to CSV');
    } catch (error) {
        console.error('Error exporting:', error);
        alert('Error exporting: ' + error.message);
    }
}

async function clearChanges() {
    if (sessionChanges.length === 0) {
        alert('No changes to clear');
        return;
    }

    if (confirm('Are you sure you want to clear all recorded changes? This will only clear the log, not revert actual changes.')) {
        sessionChanges = [];
        await fetch('/api/changes/clear', { method: 'POST' });
        updateChangesCount();
        alert('✅ Changes cleared');
    }
}

// Store current videos for reference
let currentVideos = [];

async function refreshVideosList() {
    try {
        console.log('[refreshVideosList] Fetching updated videos...');
        const response = await fetch('/api/videos');
        if (!response.ok) {
            console.error('[refreshVideosList] Failed to refresh videos list:', response.status);
            return;
        }
        const videos = await response.json();
        console.log('[refreshVideosList] Received', videos.length, 'videos');
        console.log('[refreshVideosList] Videos with relabel_history:', videos.filter(v => v.relabel_history).map(v => v.name));
        const filteredVideos = videos.filter(v => v.annotated_frames > 0);
        currentVideos = filteredVideos;
        displayVideos(filteredVideos);
        
        // Re-select the current video to maintain context
        if (currentVideo) {
            selectVideo(currentVideo);
        }
    } catch (error) {
        console.error('[refreshVideosList] Error refreshing videos list:', error);
    }
}

function showReviewHistory(videoName) {
    const video = currentVideos.find(v => v.name === videoName);
    
    if (!video || !video.relabel_history) {
        alert('No review history for this video');
        return;
    }
    
    const modal = document.getElementById('review-history-modal');
    const historyContent = document.getElementById('history-content');
    
    let html = `<div style="margin-bottom: 1rem;"><strong>Video:</strong> ${videoName}</div>`;
    html += `<div style="margin-bottom: 1.5rem;"><strong>Reviewed by:</strong> ${video.relabel_history.reviewers.join(', ')}</div>`;
    html += `<h4 style="margin-bottom: 1rem;">Changes Made:</h4>`;
    
    if (video.relabel_history.changes.length === 0) {
        html += '<p>No changes recorded.</p>';
    } else {
        video.relabel_history.changes.forEach(change => {
            html += `
                <div class="history-item">
                    <div class="history-item-header">
                        <span class="history-annotator">👤 ${change.annotator}</span>
                        <span class="history-timestamp">${new Date(change.timestamp).toLocaleString()}</span>
                    </div>
                    <div class="history-change">
                        <span style="background: #ff6b6b; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold;">${change.old_class}</span>
                        <span class="change-arrow">→</span>
                        <span style="background: #51cf66; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold;">${change.new_class}</span>
                    </div>
                </div>
            `;
        });
    }
    
    historyContent.innerHTML = html;
    modal.style.display = 'flex';
}

async function toggleVideoFlag(videoName, flagged, checkboxElement) {
    if (!currentAnnotator) {
        alert('⚠️ Please select an annotator name first');
        document.getElementById('annotator-select').focus();
        // Reset checkbox
        if (checkboxElement) {
            checkboxElement.checked = !flagged;
        }
        return;
    }
    
    try {
        const response = await fetch(`/api/video/${videoName}/flag`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                flagged: flagged,
                annotator: currentAnnotator,
                reason: flagged ? 'Bounding boxes need review' : ''
            })
        });
        
        if (!response.ok) {
            throw new Error(`Failed to toggle flag: ${response.status}`);
        }
        
        const result = await response.json();
        console.log(`Video ${videoName} ${flagged ? 'flagged' : 'unflagged'} by ${currentAnnotator}`);
        
        // Refresh video list to update UI
        await refreshVideosList();
    } catch (error) {
        console.error('Error toggling flag:', error);
        alert('Error toggling flag: ' + error.message);
        // Reset checkbox on error
        if (checkboxElement) {
            checkboxElement.checked = !flagged;
        }
    }

    reviewBadges.forEach(badge => {
        badge.style.cursor = 'pointer';
        badge.addEventListener('click', (e) => {
            e.stopPropagation();
            const videoItem = badge.closest('.video-item');
            const videoName = videoItem.querySelector('.video-name').textContent.trim().split('\n')[0];
            showReviewHistory(videoName);
        });
    });
}
