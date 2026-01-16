/**
 * Dashboard JavaScript - Load and display dataset statistics and charts
 */

document.addEventListener('DOMContentLoaded', async () => {
    try {
        console.log('Loading dashboard data...');
        
        // Load all data with better error handling
        const statsResponse = await fetch('/api/stats');
        console.log('Stats response status:', statsResponse.status);
        
        if (!statsResponse.ok) {
            const errorText = await statsResponse.text();
            throw new Error(`Stats API error: ${statsResponse.status} - ${errorText}`);
        }
        
        const stats = await statsResponse.json();
        console.log('Stats loaded:', stats);
        
        const videosResponse = await fetch('/api/videos');
        console.log('Videos response status:', videosResponse.status);
        
        if (!videosResponse.ok) {
            const errorText = await videosResponse.text();
            throw new Error(`Videos API error: ${videosResponse.status} - ${errorText}`);
        }
        
        const videos = await videosResponse.json();
        console.log('Videos loaded:', videos.length, 'videos');
        
        // Update header stats
        updateHeaderStats(stats);
        
        // Create all visualizations
        createClassDistributionChart(stats);
        createVideosPerClassChart(stats);
        createAnnotationsPieChart(stats);
        createFramesPerVideoChart(videos);
        createClassPresenceHeatmap(stats, videos);
        
        // Hide loading overlay
        document.getElementById('loading-overlay').style.display = 'none';
        console.log('Dashboard loaded successfully!');
    } catch (error) {
        console.error('Error loading dashboard:', error);
        document.getElementById('loading-overlay').innerHTML = `
            <div style="text-align: center; color: #ef4444;">
                <p style="font-size: 18px; font-weight: bold;">❌ Error loading dataset</p>
                <p>${error.message}</p>
                <p style="font-size: 12px; margin-top: 10px;">Check the console for details</p>
                <button onclick="location.reload()" style="margin-top: 20px; padding: 10px 20px; background: #3b82f6; color: white; border: none; border-radius: 5px; cursor: pointer;">Retry</button>
            </div>
        `;
    }
});

function updateHeaderStats(stats) {
    const elements = {
        'stat-total-images': stats.total_images,
        'stat-annotated': stats.total_annotated,
        'stat-coverage': stats.coverage.toFixed(1) + '%',
        'stat-targets': stats.total_targets,
        'stat-videos': stats.total_videos,
        'stat-avg-targets': stats.avg_targets_per_image.toFixed(2)
    };
    
    Object.entries(elements).forEach(([id, value]) => {
        document.getElementById(id).textContent = value;
    });
}

function createClassDistributionChart(stats) {
    const classes = Object.keys(stats.class_counts).sort();
    const counts = classes.map(c => stats.class_counts[c]);
    
    const colors = ['#ef4444', '#10b981', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899', '#6b7280'];
    const trace = {
        x: classes,
        y: counts,
        type: 'bar',
        marker: {
            color: colors.slice(0, classes.length)
        }
    };
    
    const layout = {
        title: 'Total Objects per Class',
        xaxis: { title: 'Class' },
        yaxis: { title: 'Count' },
        margin: { t: 40, b: 80, l: 60, r: 20 },
        showlegend: false,
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#f9fafb'
    };
    
    Plotly.newPlot('chart-class-distribution', [trace], layout, { responsive: true });
}

function createVideosPerClassChart(stats) {
    const classes = Object.keys(stats.videos_with_class).sort();
    const counts = classes.map(c => stats.videos_with_class[c]);
    
    const colors = ['#ef4444', '#10b981', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899', '#6b7280'];
    const trace = {
        x: classes,
        y: counts,
        type: 'bar',
        marker: {
            color: colors.slice(0, classes.length)
        }
    };
    
    const layout = {
        title: 'Number of Videos Containing Each Class',
        xaxis: { title: 'Class' },
        yaxis: { title: 'Video Count' },
        margin: { t: 40, b: 80, l: 60, r: 20 },
        showlegend: false,
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#f9fafb'
    };
    
    Plotly.newPlot('chart-videos-per-class', [trace], layout, { responsive: true });
}

function createAnnotationsPieChart(stats) {
    const trace = {
        labels: ['Frames with Targets', 'Background Frames'],
        values: [stats.total_annotated, stats.total_images - stats.total_annotated],
        type: 'pie',
        marker: {
            colors: ['#10b981', '#ef4444']
        },
        textposition: 'inside',
        textinfo: 'label+percent'
    };
    
    const layout = {
        title: 'Annotated vs Background Frames',
        height: 400,
        margin: { t: 40, b: 20, l: 20, r: 20 },
        paper_bgcolor: '#ffffff',
        font: { color: '#1f2937' }
    };
    
    Plotly.newPlot('chart-annotations-pie', [trace], layout, { responsive: true });
}

function createFramesPerVideoChart(videos) {
    // Sort videos by frame count (descending) and show all
    const sortedVideos = videos.sort((a, b) => b.total_frames - a.total_frames);
    
    const names = sortedVideos.map(v => v.name);
    const annotated = sortedVideos.map(v => v.annotated_frames);
    const background = sortedVideos.map(v => v.background_frames);
    
    const trace1 = {
        x: names,
        y: annotated,
        name: 'Annotated',
        type: 'bar',
        marker: { color: '#10b981' }
    };
    
    const trace2 = {
        x: names,
        y: background,
        name: 'Background',
        type: 'bar',
        marker: { color: '#ef4444' }
    };
    
    const layout = {
        title: 'Frames per Video',
        xaxis: { title: 'Video' },
        yaxis: { title: 'Frame Count' },
        barmode: 'stack',
        height: 400,
        margin: { t: 40, b: 100, l: 60, r: 20 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#f9fafb'
    };
    
    Plotly.newPlot('chart-frames-per-video', [trace1, trace2], layout, { responsive: true });
}

function createClassPresenceHeatmap(stats, videos) {
    const videoNames = videos.map(v => v.name).sort();
    const classes = Object.keys(stats.class_counts).sort();
    
    // Build presence matrix using the video names list for each class
    const presenceMatrix = [];
    for (let video of videoNames) {
        const row = [];
        for (let cls of classes) {
            // Check if this video is in the list of videos for this class
            const videoList = stats.videos_with_class_names[cls] || [];
            row.push(videoList.includes(video) ? 1 : 0);
        }
        presenceMatrix.push(row);
    }
    
    const trace = {
        z: presenceMatrix,
        x: classes,
        y: videoNames,
        type: 'heatmap',
        colorscale: [
            [0, '#f9fafb'],
            [1, '#10b981']
        ],
        showscale: false,
        hovertemplate: 'Video: %{y}<br>Class: %{x}<br>Present: %{z}<extra></extra>'
    };
    
    const layout = {
        title: 'Class Presence per Video',
        xaxis: { title: 'Class' },
        yaxis: { title: 'Video' },
        height: Math.max(300, videoNames.length * 15 + 100),
        margin: { t: 40, b: 80, l: 150, r: 20 },
        paper_bgcolor: '#ffffff'
    };
    
    Plotly.newPlot('chart-class-heatmap', [trace], layout, { responsive: true });
}
