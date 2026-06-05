/**
 * Driver Alertness Monitoring Dashboard
 * Real-time webcam feed with facial landmark detection and alertness analysis
 */

class DashboardManager {
    constructor() {
        this.isMonitoring = false;
        this.webcam = null;
        this.canvas = null;
        this.ctx = null;
        this.video = null;
        this.frameCount = 0;
        this.lastFrameTime = Date.now();
        this.fps = 0;

        // Data buffers (last 60 seconds of data)
        this.earHistory = [];
        this.confidenceHistory = [];
        this.perclosHistory = [];
        this.timestamps = [];
        this.maxHistorySize = 300; // 60s @ 5Hz

        // Statistics
        this.sessionStart = null;
        this.alertCount = 0;
        this.confidenceValues = [];
        this.perclosValues = [];
        this.lastAlertStatus = 'idle';

        // Chart instances
        this.earChart = null;
        this.confidenceChart = null;

        // Thresholds
        this.earThreshold = 0.2;
        this.drowsyThreshold = 0.3;
        this.microsleepThreshold = 3; // frames below earThreshold

        this.initializeUI();
    }

    initializeUI() {
        this.video = document.getElementById('webcam');
        this.canvas = document.getElementById('overlay');
        this.ctx = this.canvas.getContext('2d');

        // UI elements
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.alertBanner = document.getElementById('alertBanner');
        this.alertStatus = document.getElementById('alertStatus');
        this.alertMessage = document.getElementById('alertMessage');
        this.fpsCounter = document.getElementById('fpsCounter');
        this.frameCounter = document.getElementById('frameCounter');

        // Metrics
        this.earLeftEl = document.getElementById('earLeft');
        this.earRightEl = document.getElementById('earRight');
        this.marEl = document.getElementById('mar');
        this.perclosEl = document.getElementById('perclos');

        // Stats
        this.durationEl = document.getElementById('sessionDuration');
        this.alertCountEl = document.getElementById('alertCount');
        this.avgConfidenceEl = document.getElementById('avgConfidence');
        this.avgPerclosEl = document.getElementById('avgPerclos');

        // Debug
        this.debugOutput = document.getElementById('debugOutput');

        // Event listeners
        this.startBtn.addEventListener('click', () => this.start());
        this.stopBtn.addEventListener('click', () => this.stop());

        // Initialize charts
        this.initializeCharts();
    }

    initializeCharts() {
        const chartConfig = {
            type: 'line',
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#a0aac0',
                            font: { family: "'Courier New', monospace" }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(58, 69, 96, 0.2)' },
                        ticks: { color: '#a0aac0' }
                    },
                    y: {
                        grid: { color: 'rgba(58, 69, 96, 0.2)' },
                        ticks: { color: '#a0aac0' }
                    }
                }
            }
        };

        // EAR Chart
        const earCtx = document.getElementById('earChart').getContext('2d');
        this.earChart = new Chart(earCtx, {
            ...chartConfig,
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Left EAR',
                        data: [],
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: true
                    },
                    {
                        label: 'Right EAR',
                        data: [],
                        borderColor: '#7c3aed',
                        backgroundColor: 'rgba(124, 58, 237, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: true
                    },
                    {
                        label: 'Threshold',
                        data: [],
                        borderColor: '#ef4444',
                        borderWidth: 1,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    }
                ]
            }
        });

        // Confidence Chart
        const confCtx = document.getElementById('confidenceChart').getContext('2d');
        this.confidenceChart = new Chart(confCtx, {
            ...chartConfig,
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Confidence',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: true
                    }
                ]
            }
        });
    }

    async start() {
        try {
            this.log('Starting monitoring...');
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });

            this.video.srcObject = stream;
            this.video.onloadedmetadata = () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
            };

            this.isMonitoring = true;
            this.sessionStart = Date.now();
            this.alertCount = 0;
            this.confidenceValues = [];
            this.perclosValues = [];
            this.earHistory = [];
            this.confidenceHistory = [];
            this.perclosHistory = [];
            this.timestamps = [];

            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;

            this.captureLoop();
            this.updateStatsLoop();
        } catch (err) {
            this.log(`Error: ${err.message}`);
            alert('Failed to access webcam: ' + err.message);
        }
    }

    stop() {
        this.isMonitoring = false;
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(t => t.stop());
        }
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.setAlertStatus('idle', 'System stopped');
        this.log('Monitoring stopped');
    }

    captureLoop = async () => {
        if (!this.isMonitoring) return;

        try {
            // Capture frame
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            const imageData = this.canvas.toDataURL('image/jpeg', 0.8);

            // Send to backend every 200ms
            if (this.frameCount % 4 === 0) {
                this.sendPredictionRequest(imageData);
            }

            // Update FPS
            this.frameCount++;
            const now = Date.now();
            const elapsed = now - this.lastFrameTime;
            if (elapsed >= 1000) {
                this.fps = (this.frameCount * 1000) / elapsed;
                this.fpsCounter.textContent = `FPS: ${this.fps.toFixed(1)}`;
                this.frameCounter.textContent = `Frames: ${this.frameCount}`;
                this.frameCount = 0;
                this.lastFrameTime = now;
            }
        } catch (err) {
            this.log(`Capture error: ${err.message}`);
        }

        // Request next frame (200ms interval)
        setTimeout(this.captureLoop, 200);
    };

    async sendPredictionRequest(imageData) {
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ data: imageData })
            });

            if (!response.ok) {
                this.log(`Server error: ${response.status}`);
                return;
            }

            const result = await response.json();
            this.processPredictionResult(result);
        } catch (err) {
            this.log(`Prediction error: ${err.message}`);
        }
    }

    processPredictionResult(result) {
        // Extract metrics from result
        const earLeft = result.ear_left || 0;
        const earRight = result.ear_right || 0;
        const mar = result.mar || 0;
        const avgEar = (earLeft + earRight) / 2;
        const confidence = result.confidence || 0;
        const perclos = result.perclos || 0;

        // Update display
        this.earLeftEl.textContent = earLeft.toFixed(3);
        this.earRightEl.textContent = earRight.toFixed(3);
        this.marEl.textContent = mar.toFixed(3);
        this.perclosEl.textContent = (perclos * 100).toFixed(1) + '%';

        // Add to history
        const timestamp = new Date().toLocaleTimeString();
        this.earHistory.push(avgEar);
        this.confidenceHistory.push(confidence);
        this.perclosHistory.push(perclos);
        this.timestamps.push(timestamp);
        this.confidenceValues.push(confidence);
        this.perclosValues.push(perclos);

        // Keep history size in check
        if (this.earHistory.length > this.maxHistorySize) {
            this.earHistory.shift();
            this.confidenceHistory.shift();
            this.perclosHistory.shift();
            this.timestamps.shift();
        }

        // Update charts
        this.updateCharts();

        // Determine alert status
        this.updateAlertStatus(avgEar, mar);

        // Draw landmarks on canvas
        this.drawOverlay(result);
    }

    updateAlertStatus(avgEar, mar) {
        let status = 'idle';
        let message = 'Alert monitoring active';

        // Check for drowsiness or microsleep
        const closedEyeCount = this.earHistory.filter(ear => ear < this.earThreshold).length;
        const recentFrames = Math.min(30, this.earHistory.length);
        const recentClosedCount = this.earHistory.slice(-recentFrames).filter(e => e < this.earThreshold).length;

        if (recentClosedCount >= this.microsleepThreshold && recentFrames >= 10) {
            status = 'microsleep';
            message = '⚠️ MICROSLEEP DETECTED - WAKE UP!';
            this.alertCount++;
        } else if (avgEar < this.drowsyThreshold) {
            status = 'drowsy';
            message = '⚠️ DROWSY - Increased Alertness Required';
        } else if (avgEar < this.earThreshold) {
            status = 'alert';
            message = '⚠️ ALERT - Eyes Closed';
            this.alertCount++;
        } else {
            status = 'normal';
            message = '✓ Normal - Eyes Open';
        }

        if (status !== this.lastAlertStatus) {
            this.setAlertStatus(status, message);
            this.lastAlertStatus = status;
        }
    }

    setAlertStatus(status, message) {
        this.alertBanner.className = `alert-banner alert-${status}`;
        this.alertStatus.textContent = status.toUpperCase();
        this.alertMessage.textContent = message;
        this.alertCountEl.textContent = this.alertCount;
    }

    updateCharts() {
        // Calculate threshold line
        const thresholdLine = new Array(this.earHistory.length).fill(this.earThreshold);

        this.earChart.data.labels = this.timestamps.slice(-60);
        this.earChart.data.datasets[0].data = this.earHistory.slice(-60);
        this.earChart.data.datasets[1].data = this.earHistory.slice(-60); // simplified for now
        this.earChart.data.datasets[2].data = thresholdLine.slice(-60);
        this.earChart.update('none');

        this.confidenceChart.data.labels = this.timestamps.slice(-60);
        this.confidenceChart.data.datasets[0].data = this.confidenceHistory.slice(-60);
        this.confidenceChart.update('none');
    }

    drawOverlay(result) {
        // Simple overlay - can be enhanced with landmarks
        this.ctx.fillStyle = 'rgba(0, 212, 255, 0.1)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw metrics text
        this.ctx.fillStyle = '#00d4ff';
        this.ctx.font = 'bold 16px "Courier New"';
        this.ctx.fillText(`EAR: ${(result.ear_left || 0).toFixed(3)}`, 10, 30);
        this.ctx.fillText(`MAR: ${(result.mar || 0).toFixed(3)}`, 10, 60);
    }

    updateStatsLoop = () => {
        if (this.isMonitoring) {
            this.updateStats();
            setTimeout(this.updateStatsLoop, 1000);
        }
    };

    updateStats() {
        if (!this.sessionStart) return;

        // Duration
        const elapsed = Math.floor((Date.now() - this.sessionStart) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        this.durationEl.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Averages
        if (this.confidenceValues.length > 0) {
            const avgConf = this.confidenceValues.reduce((a, b) => a + b) / this.confidenceValues.length;
            this.avgConfidenceEl.textContent = (avgConf * 100).toFixed(1) + '%';
        }

        if (this.perclosValues.length > 0) {
            const avgPerclos = this.perclosValues.reduce((a, b) => a + b) / this.perclosValues.length;
            this.avgPerclosEl.textContent = (avgPerclos * 100).toFixed(1) + '%';
        }
    }

    log(message) {
        const time = new Date().toLocaleTimeString();
        const line = `[${time}] ${message}`;
        this.debugOutput.textContent = line + '\n' + this.debugOutput.textContent;
        // Keep debug output limited
        const lines = this.debugOutput.textContent.split('\n');
        if (lines.length > 20) {
            this.debugOutput.textContent = lines.slice(0, 20).join('\n');
        }
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new DashboardManager();
});
