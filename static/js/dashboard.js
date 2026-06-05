/**
 * Driver Alertness Monitoring Dashboard
 * Real-time webcam feed with facial landmark detection and alertness analysis
 */

class DashboardManager {
    constructor() {
        this.isMonitoring = false;
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.drawRaf = null;
        this.lastResult = null;
        this.frameCount = 0;
        this.lastFrameTime = Date.now();
        this.fps = 0;

        // Data buffers
        this.timestamps = [];          // Date.now() values
        this.earHistory = [];          // avgEar — used for alert logic
        this.earLeftHistory = [];
        this.earRightHistory = [];
        this.confidenceHistory = [];
        this.perclosHistory = [];
        this.maxHistorySize = 300;

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
        this.microsleepThreshold = 3;

        this.initializeUI();
    }

    initializeUI() {
        this.video = null; // created dynamically in start()
        this.canvas = document.getElementById('overlay');
        this.ctx = this.canvas.getContext('2d');

        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.alertBanner = document.getElementById('alertBanner');
        this.alertStatus = document.getElementById('alertStatus');
        this.alertMessage = document.getElementById('alertMessage');
        this.fpsCounter = document.getElementById('fpsCounter');
        this.frameCounter = document.getElementById('frameCounter');

        this.earLeftEl = document.getElementById('earLeft');
        this.earRightEl = document.getElementById('earRight');
        this.marEl = document.getElementById('mar');
        this.perclosEl = document.getElementById('perclos');

        this.durationEl = document.getElementById('sessionDuration');
        this.alertCountEl = document.getElementById('alertCount');
        this.avgConfidenceEl = document.getElementById('avgConfidence');
        this.avgPerclosEl = document.getElementById('avgPerclos');

        this.debugOutput = document.getElementById('debugOutput');

        this.startBtn.addEventListener('click', () => this.start());
        this.stopBtn.addEventListener('click', () => this.stop());

        this.initializeCharts();
    }

    initializeCharts() {
        const now = Date.now();

        const xAxisConfig = {
            type: 'linear',
            min: now,
            max: now + 60000,
            grid: { color: 'rgba(58, 69, 96, 0.2)' },
            ticks: {
                color: '#a0aac0',
                maxTicksLimit: 6,
                callback: (value) => {
                    const d = new Date(value);
                    return d.getHours().toString().padStart(2, '0') + ':' +
                           d.getMinutes().toString().padStart(2, '0') + ':' +
                           d.getSeconds().toString().padStart(2, '0');
                }
            }
        };

        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#a0aac0',
                        font: { family: "'Courier New', monospace" }
                    }
                }
            }
        };

        const earCtx = document.getElementById('earChart').getContext('2d');
        this.earChart = new Chart(earCtx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Left EAR',
                        data: [],
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.3,
                        fill: true
                    },
                    {
                        label: 'Right EAR',
                        data: [],
                        borderColor: '#7c3aed',
                        backgroundColor: 'rgba(124, 58, 237, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.3,
                        fill: true
                    },
                    {
                        label: 'Threshold',
                        data: [],
                        borderColor: '#ef4444',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        tension: 0,
                        fill: false
                    }
                ]
            },
            options: {
                ...commonOptions,
                scales: {
                    x: xAxisConfig,
                    y: {
                        min: 0,
                        max: 0.5,
                        grid: { color: 'rgba(58, 69, 96, 0.2)' },
                        ticks: { color: '#a0aac0' }
                    }
                }
            }
        });

        const confCtx = document.getElementById('confidenceChart').getContext('2d');
        this.confidenceChart = new Chart(confCtx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Confidence',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.3,
                        fill: true
                    }
                ]
            },
            options: {
                ...commonOptions,
                scales: {
                    x: xAxisConfig,
                    y: {
                        min: 0,
                        max: 1.0,
                        grid: { color: 'rgba(58, 69, 96, 0.2)' },
                        ticks: { color: '#a0aac0' }
                    }
                }
            }
        });
    }

    async start() {
        try {
            this.log('Starting monitoring...');

            // Create video element in JS — does not need to be in the DOM
            const video = document.createElement('video');
            video.autoplay = true;
            video.playsInline = true;
            video.muted = true;
            this.video = video;

            navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
                video.srcObject = stream;
                video.play();
                video.addEventListener('playing', () => {
                    this.canvas.width = video.videoWidth || 640;
                    this.canvas.height = video.videoHeight || 480;
                    this.log(`Video playing at ${this.canvas.width}x${this.canvas.height}`);

                    const drawFrame = () => {
                        if (!this.isMonitoring) return;
                        this.ctx.drawImage(video, 0, 0, this.canvas.width, this.canvas.height);
                        if (this.lastResult) this.drawOverlay(this.lastResult);
                        this.drawRaf = requestAnimationFrame(drawFrame);
                    };
                    drawFrame();
                });
            }).catch(err => {
                this.log(`Camera error: ${err.message}`);
                alert('Failed to access webcam: ' + err.message);
            });

            this.isMonitoring = true;
            this.sessionStart = Date.now();
            this.alertCount = 0;
            this.confidenceValues = [];
            this.perclosValues = [];
            this.earHistory = [];
            this.earLeftHistory = [];
            this.earRightHistory = [];
            this.confidenceHistory = [];
            this.perclosHistory = [];
            this.timestamps = [];
            this.lastResult = null;

            // Reset chart x-axis window to start from now
            const now = Date.now();
            this.earChart.options.scales.x.min = now;
            this.earChart.options.scales.x.max = now + 60000;
            this.confidenceChart.options.scales.x.min = now;
            this.confidenceChart.options.scales.x.max = now + 60000;
            this.earChart.data.datasets.forEach(ds => { ds.data = []; });
            this.confidenceChart.data.datasets.forEach(ds => { ds.data = []; });

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
        if (this.drawRaf) {
            cancelAnimationFrame(this.drawRaf);
            this.drawRaf = null;
        }
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(t => t.stop());
            this.video.srcObject = null;
        }
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.setAlertStatus('idle', 'System stopped');
        this.log('Monitoring stopped');
    }

    // Runs at 5 Hz — captures the current canvas frame and sends to the backend
    captureLoop = async () => {
        if (!this.isMonitoring) return;

        try {
            const imageData = this.canvas.toDataURL('image/jpeg', 0.8);

            if (this.frameCount % 4 === 0) {
                this.sendPredictionRequest(imageData);
            }

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
        const earLeft = result.ear_left || 0;
        const earRight = result.ear_right || 0;
        const mar = result.mar || 0;
        const avgEar = (earLeft + earRight) / 2;
        const confidence = result.confidence || 0;
        const perclos = result.perclos || 0;

        this.earLeftEl.textContent = earLeft.toFixed(3);
        this.earRightEl.textContent = earRight.toFixed(3);
        this.marEl.textContent = mar.toFixed(3);
        this.perclosEl.textContent = (perclos * 100).toFixed(1) + '%';

        const ts = Date.now();
        this.timestamps.push(ts);
        this.earHistory.push(avgEar);
        this.earLeftHistory.push(earLeft);
        this.earRightHistory.push(earRight);
        this.confidenceHistory.push(confidence);
        this.perclosHistory.push(perclos);
        this.confidenceValues.push(confidence);
        this.perclosValues.push(perclos);

        if (this.timestamps.length > this.maxHistorySize) {
            this.timestamps.shift();
            this.earHistory.shift();
            this.earLeftHistory.shift();
            this.earRightHistory.shift();
            this.confidenceHistory.shift();
            this.perclosHistory.shift();
        }

        this.lastResult = result;
        this.updateCharts();
        this.updateAlertStatus(avgEar, mar);
    }

    updateAlertStatus(avgEar, mar) {
        let status = 'idle';
        let message = 'Alert monitoring active';

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
        const now = Date.now();
        const windowMs = 60000;

        // Slide the x-axis window so recent data is always visible
        this.earChart.options.scales.x.min = now - windowMs;
        this.earChart.options.scales.x.max = now;
        this.confidenceChart.options.scales.x.min = now - windowMs;
        this.confidenceChart.options.scales.x.max = now;

        // Build {x, y} point arrays for linear axis
        const earLeftData  = this.timestamps.map((ts, i) => ({ x: ts, y: this.earLeftHistory[i] }));
        const earRightData = this.timestamps.map((ts, i) => ({ x: ts, y: this.earRightHistory[i] }));
        const thresholdData = [
            { x: now - windowMs, y: this.earThreshold },
            { x: now,            y: this.earThreshold }
        ];
        const confData = this.timestamps.map((ts, i) => ({ x: ts, y: this.confidenceHistory[i] }));

        this.earChart.data.datasets[0].data = earLeftData;
        this.earChart.data.datasets[1].data = earRightData;
        this.earChart.data.datasets[2].data = thresholdData;
        this.earChart.update('none');

        this.confidenceChart.data.datasets[0].data = confData;
        this.confidenceChart.update('none');
    }

    drawOverlay(result) {
        // Small dark box so the text is legible over any video background
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        this.ctx.fillRect(5, 5, 150, 65);
        this.ctx.fillStyle = '#00d4ff';
        this.ctx.font = 'bold 14px "Courier New"';
        this.ctx.fillText(`EAR: ${(result.ear_left || 0).toFixed(3)}`, 12, 28);
        this.ctx.fillText(`MAR: ${(result.mar || 0).toFixed(3)}`, 12, 52);
    }

    updateStatsLoop = () => {
        if (this.isMonitoring) {
            this.updateStats();
            setTimeout(this.updateStatsLoop, 1000);
        }
    };

    updateStats() {
        if (!this.sessionStart) return;

        const elapsed = Math.floor((Date.now() - this.sessionStart) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        this.durationEl.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

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
