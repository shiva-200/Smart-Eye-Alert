// Constants
const EYE_ASPECT_RATIO_THRESHOLD = 0.3;
const EYE_ASPECT_RATIO_CONSEC_FRAMES = 50;
let counter = 0;
let isDetectionRunning = false;

// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusElement = document.getElementById('status');
const earValueElement = document.getElementById('ear-value');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const alertSound = document.getElementById('alert');

// Load face-api.js models
async function loadModels() {
    try {
        // Load models from CDN
        await faceapi.nets.tinyFaceDetector.loadFromUri('https://justadudewhohacks.github.io/face-api.js/models');
        await faceapi.nets.faceLandmark68TinyNet.loadFromUri('https://justadudewhohacks.github.io/face-api.js/models');
        console.log('Models loaded successfully');
    } catch (error) {
        console.error('Error loading models:', error);
        alert('Error loading face detection models. Please check the console for details.');
    }
}

// Calculate Eye Aspect Ratio
function calculateEAR(eye) {
    const A = Math.hypot(
        eye[1].x - eye[5].x,
        eye[1].y - eye[5].y
    );
    const B = Math.hypot(
        eye[2].x - eye[4].x,
        eye[2].y - eye[4].y
    );
    const C = Math.hypot(
        eye[0].x - eye[3].x,
        eye[0].y - eye[3].y
    );
    return (A + B) / (2.0 * C);
}

// Process video frame
async function processFrame() {
    if (!isDetectionRunning) return;

    try {
        const detections = await faceapi.detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks(true);

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        if (detections) {
            const landmarks = detections.landmarks;
            const leftEye = landmarks.getLeftEye();
            const rightEye = landmarks.getRightEye();

            const leftEAR = calculateEAR(leftEye);
            const rightEAR = calculateEAR(rightEye);
            const ear = (leftEAR + rightEAR) / 2.0;

            earValueElement.textContent = `EAR: ${ear.toFixed(2)}`;

            // Draw eye contours
            ctx.strokeStyle = '#00FF00';
            ctx.lineWidth = 2;
            drawEyeContour(leftEye);
            drawEyeContour(rightEye);

            if (ear < EYE_ASPECT_RATIO_THRESHOLD) {
                counter++;
                if (counter >= EYE_ASPECT_RATIO_CONSEC_FRAMES) {
                    statusElement.textContent = 'Status: DROWSY ALERT!';
                    statusElement.classList.add('alert');
                    if (alertSound.paused) {
                        alertSound.play();
                    }
                }
            } else {
                counter = 0;
                statusElement.textContent = 'Status: Monitoring...';
                statusElement.classList.remove('alert');
                alertSound.pause();
                alertSound.currentTime = 0;
            }
        }
    } catch (error) {
        console.error('Error processing frame:', error);
    }

    requestAnimationFrame(processFrame);
}

function drawEyeContour(eye) {
    ctx.beginPath();
    ctx.moveTo(eye[0].x, eye[0].y);
    for (let i = 1; i < eye.length; i++) {
        ctx.lineTo(eye[i].x, eye[i].y);
    }
    ctx.closePath();
    ctx.stroke();
}

// Start video stream
async function startVideo() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        await video.play();
        
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        isDetectionRunning = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        
        processFrame();
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Error accessing camera. Please ensure you have granted camera permissions.');
    }
}

// Stop video stream
function stopVideo() {
    const stream = video.srcObject;
    const tracks = stream.getTracks();
    
    tracks.forEach(track => track.stop());
    video.srcObject = null;
    
    isDetectionRunning = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    statusElement.textContent = 'Status: Stopped';
    statusElement.classList.remove('alert');
    alertSound.pause();
    alertSound.currentTime = 0;

    // Reload the page after a short delay
    setTimeout(() => {
        window.location.reload();
    }, 500);
}

// Event Listeners
startBtn.addEventListener('click', startVideo);
stopBtn.addEventListener('click', stopVideo);

// Initialize
loadModels(); 