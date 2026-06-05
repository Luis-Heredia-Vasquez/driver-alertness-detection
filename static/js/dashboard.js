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

        // earHistory kept for alert logic; charts manage their own labels/data
        this.earHistory = [];
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
        const commonOptions = {
            animation: false,
            responsive: true,
            maintainAspectRatio: false,
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
                labels: [],
                datasets: [
                    {
                        label: 'Left EAR',
                        data: [],
                        borderColor: '#00d4ff',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.4,
                        fill: false
                    },
                    {
                        label: 'Right EAR',
                        data: [],
                        borderColor: '#7c3aed',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.4,
                        fill: false
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
                    x: { display: false },
                    y: { min: 0, max: 0.5 }
                }
            }
        });

        const confCtx = document.getElementById('confidenceChart').getContext('2d');
        this.confidenceChart = new Chart(confCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Confidence',
                        data: [],
                        borderColor: '#10b981',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.4,
                        fill: false
                    }
                ]
            },
            options: {
                ...commonOptions,
                scales: {
                    x: { display: false },
                    y: { min: 0, max: 1.0 }
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
            this.lastResult = null;

            // Clear chart data for fresh session
            this.earChart.data.labels = [];
            this.earChart.data.datasets.forEach(ds => { ds.data = []; });
            this.confidenceChart.data.labels = [];
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

        this.earHistory.push(avgEar);
        this.confidenceValues.push(confidence);
        this.perclosValues.push(perclos);
        if (this.earHistory.length > this.maxHistorySize) {
            this.earHistory.shift();
        }

        this.lastResult = result;
        this.updateCharts(earLeft, earRight, confidence);
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

    updateCharts(earLeft, earRight, confidence) {
        const label = new Date().toLocaleTimeString();

        // EAR chart — push new values, keep max 60 entries
        this.earChart.data.labels.push(label);
        this.earChart.data.datasets[0].data.push(earLeft);
        this.earChart.data.datasets[1].data.push(earRight);
        this.earChart.data.datasets[2].data.push(this.earThreshold);
        if (this.earChart.data.labels.length > 60) {
            this.earChart.data.labels.shift();
            this.earChart.data.datasets[0].data.shift();
            this.earChart.data.datasets[1].data.shift();
            this.earChart.data.datasets[2].data.shift();
        }
        this.earChart.update();

        // Confidence chart — same pattern
        this.confidenceChart.data.labels.push(label);
        this.confidenceChart.data.datasets[0].data.push(confidence);
        if (this.confidenceChart.data.labels.length > 60) {
            this.confidenceChart.data.labels.shift();
            this.confidenceChart.data.datasets[0].data.shift();
        }
        this.confidenceChart.update();
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
