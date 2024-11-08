let detectionSocket;
let cart = [];
let isProcessing = false;

/**
 * Entry point to setup POS system
 * Sets up the cameras and calls other functions
 */
async function setupCameras() {
    try {
        // Get list of video devices
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');

        // Setup detection view camera
        const detectionStream = await navigator.mediaDevices.getUserMedia({
            video: { 
                deviceId: { exact: videoDevices[1].deviceId },
                // Request 4K resolution
                width: { ideal: 3840 },
                height: { ideal: 2160 }
            }
        });
        const detectionVideo = document.createElement('video');
        detectionVideo.srcObject = detectionStream;
        detectionVideo.play();

        // Setup customer view camera
        const customerStream = await navigator.mediaDevices.getUserMedia({
            video: { 
                deviceId: { exact: videoDevices[0].deviceId },
            }
        });
        const customerVideo = document.createElement('video');
        customerVideo.srcObject = customerStream;
        customerVideo.play();

        // Setup canvas for detection view
        const detectionView = document.getElementById('detectionView');
        const detectionCtx = detectionView.getContext('2d');

        // Setup canvas for customer view
        const customerView = document.getElementById('customerView');
        const customerCtx = customerView.getContext('2d');

        // Setup WebSocket connection for python detection backend
        setupWebSocket();

        // Start render loops
        renderDetectionView(detectionView, detectionCtx, detectionVideo);
        renderCustomerView(customerView, customerCtx, customerVideo);

    } catch (error) {
        console.error('Error setting up cameras:', error);
    }
}

/**
 * Setup WebSocket connection to the detection backend
 */
function setupWebSocket() {
    // Initialise websocket connection
    detectionSocket = new WebSocket('ws://localhost:8000/ws');

    // Start the frame processing loop once WebSocket is open
    detectionSocket.onopen = () => {
        console.log('WebSocket connected');
        sendNextFrame();
    };

    // Get new detections from the response
    detectionSocket.onmessage = (event) => {
        const newDetections = JSON.parse(event.data);

        // Update the cart with the new detections
        updateCart(newDetections);
        
        // Reset the processing flag
        isProcessing = false;

        // After processing the response, send the next frame
        sendNextFrame();
    };

    // Attempt to reconnect if WebSocket is closed
    detectionSocket.onclose = () => {
        console.log('WebSocket disconnected');
        setTimeout(setupWebSocket, 1000);
    };
}

/**
 * Send next frame for processing
 */
function sendNextFrame() {
    // Don't send a frame if already processing or WebSocket isn't open
    if (isProcessing || detectionSocket?.readyState !== WebSocket.OPEN) {
        return;
    }

    // Set processing flag to stop sending multiple frames
    isProcessing = true;

    // Get the current frame from the detection view
    const detectionView = document.getElementById('detectionView');
    const frameData = detectionView.toDataURL('image/jpeg', 0.8);

    // Send the frame to the detection backend
    detectionSocket.send(frameData);
}

/**
 * Update the cart with new detections
 * Update cart display in UI
 * @param {*} newDetections Array of new detections from backend
 */
function updateCart(newDetections) {
    // Update the cart with the new detections
    cart = newDetections;

    // Get cart component
    const cartItemsLayout = document.querySelector('.cart-items');

    // Clear the cart before updating
    cartItemsLayout.innerHTML = '';

    // Reset total price counter
    let total = 0;

    // Add each item to the cart display
    cart.forEach(item => {
        // Initialise new row
        const row = document.createElement('div');
        row.className = 'cart-item';

        // Add item name
        const itemName = document.createElement('div');
        itemName.className = 'item-name';
        itemName.textContent = item.item;

        // Add item price
        const itemPrice = document.createElement('div');
        itemPrice.className = 'item-price';
        itemPrice.textContent = `$${parseFloat(item.price).toFixed(2)}`;

        // Append row to cart
        row.appendChild(itemName);
        row.appendChild(itemPrice);
        cartItemsLayout.appendChild(row);

        // Add item price to total
        total += parseFloat(item.price);
    });

    // Update checkout button
    const checkoutButton = document.getElementById('checkoutButton');
    checkoutButton.textContent = `CHECKOUT: $${total.toFixed(2)}`;
}

/**
 * Render detection view canvas
 * @param {*} detectionView Detection view canvas
 * @param {*} detectionCtx Detection view canvas context
 * @param {*} detectionVideo Detection video element
 */
function renderDetectionView(detectionView, detectionCtx, detectionVideo) {
    function render() {
        // Get current canvas width and height (which matches detectionView size)
        const canvasWidth = detectionView.width;
        const canvasHeight = detectionView.height;

        // Draw the video frame onto the canvas, scaling it down to the canvas size
        detectionCtx.drawImage(detectionVideo, 0, 0, canvasWidth, canvasHeight);

        // Draw annotations (scaled to the canvas size)
        drawAnnotations(detectionCtx, cart);

        // Trigger sending the next frame
        sendNextFrame();

        requestAnimationFrame(render);
    }
    render();
}

/**
 * Draw annotation on detection view canvas
 * @param {*} ctx Detection view canvas context
 * @param {*} cart Array of items detected
 * @param {*} canvasWidth Width of detection view canvas
 * @param {*} canvasHeight Height of detection view canvas
 */
function drawAnnotations(ctx, cart) {
    // Draw bounding boxes and item names for each item in the cart
    cart.forEach(item => {
        const [x1, y1, x2, y2] = item.bbox;

        // Draw the bounding box on the canvas with the scaled coordinates
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 5;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Draw the item name
        ctx.fillStyle = '#00ff00';
        ctx.font = '64px Arial';
        ctx.fillText(item.item, x1, y1 - 5);
    });
}

/**
 * Render customer view canvas
 * @param {*} customerView Customer view canvas
 * @param {*} customerCtx Customer view canvas context
 * @param {*} customerVideo Customer video element
 */
function renderCustomerView(customerView, customerCtx, customerVideo) {
    function render() {
        customerCtx.drawImage(customerVideo, 0, 0, customerView.width, customerView.height);
        requestAnimationFrame(render);
    }
    render();
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', setupCameras);
