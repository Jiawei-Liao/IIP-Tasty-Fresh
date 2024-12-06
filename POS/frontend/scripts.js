let detectionSocket;
let cart = [];
let isProcessing = false;
let product_list = [];

// Video elements for detection and customer views
let detectionImage;
let newDetectionImage;
let detectionVideo;
let customerVideo;

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
                deviceId: { exact: videoDevices[0].deviceId },
                // Request 4K resolution
                
                width: { ideal: 1920 },
                height: { ideal: 1080 }
                /*
                width: { ideal: 3840 },
                height: { ideal: 2160 }
                */
            }
        });
        detectionVideo = document.createElement('video');
        detectionVideo.srcObject = detectionStream;
        detectionVideo.play();

        // Setup customer view camera
        const customerStream = await navigator.mediaDevices.getUserMedia({
            video: { 
                deviceId: { exact: videoDevices[0].deviceId },
            }
        });
        customerVideo = document.createElement('video');
        customerVideo.srcObject = customerStream;
        customerVideo.play();

        // Setup canvas for detection view
        const detectionView = document.getElementById('detectionView');
        const detectionCtx = detectionView.getContext('2d');

        // Setup canvas for customer view
        const customerView = document.getElementById('customerView');
        const customerCtx = customerView.getContext('2d');
        
        // Load product list
        loadProductList();

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
 * Load product list from XLSX file
 */
async function loadProductList() {
    try {
        const response = await fetch('http://localhost:8000/products');
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        product_list = await response.json();
        console.log('Product catalog loaded:', product_list);

    } catch (error) {
        console.error('Error loading product list:', error);
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
    detectionSocket.onmessage = async (event) => {
        // Wait for customer to finish transaction before processing next frame
        await customerTransaction();

        // Parse the new detections from the response
        const newDetections = JSON.parse(event.data);

        // Update the cart with the new detections
        updateCart(newDetections);

        // Update detection image with new detection image
        detectionImage = newDetectionImage;
        
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
 * Pause detection when customer is making a transaction
 * @returns Promise that resolves when customer transaction is complete
 */
function customerTransaction() {
    return new Promise(resolve => {
        const checkFlag = () => {
            // Resolve if customer transaction is complete
            if (!checkoutModalsVisible()) {
                resolve();
            // Check again after 500ms
            } else {
                setTimeout(checkFlag, 500);
            }
        };
        checkFlag();
    });
}

/**
 * Checks if any of the checkout modals are visible
 * @returns true if any checkout modals are visible
 */
function checkoutModalsVisible() {
    const modals = [
        document.getElementById('verificationModal'),
        document.getElementById('reviewCartModal'),
        document.getElementById('thankYouModal')
    ];

    return modals.some(modal => modal.style.display === 'block');
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

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Set canvas size to match detection view size
    canvas.width = detectionView.width;
    canvas.height = detectionView.height;

    // Draw the current video frame onto the canvas
    ctx.drawImage(detectionVideo, 0, 0, canvas.width, canvas.height);

    // Temporary store the new detection image
    newDetectionImage = canvas;

    // Send the frame to the detection backend
    const frameData = canvas.toDataURL('image/jpeg', 0.8);
    detectionSocket.send(frameData);
}

/**
 * Update the cart with new detections
 * Update cart display in UI
 * @param {Array} newDetections Array of new detections from backend
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
        itemName.textContent = item.display_name;

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
 * @param {HTMLCanvasElement} detectionView Detection view canvas
 * @param {CanvasRenderingContext2D} detectionCtx Detection view canvas context
 * @param {HTMLVideoElement} detectionVideo Detection video element
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
 * @param {CanvasRenderingContext2D} ctx Canvas context to draw on
 * @param {Array} cart Array of detected items with their bounding boxes
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
        ctx.fillText(item.display_name, x1, y1 - 5);
    });
}

/**
 * Render customer view canvas
 * @param {HTMLCanvasElement} customerView Canvas element for customer view
 * @param {CanvasRenderingContext2D} customerCtx Canvas context
 * @param {HTMLVideoElement} customerVideo Video element for customer feed
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

// Checkout variables
let currentItemIndex = 0;
let checkoutCart = [];

// Add checkout process functions when checkout button is clicked
document.getElementById('checkoutButton').addEventListener('click', () => {
    checkoutCart = cart.map(item => ({ ...item }));
    currentItemIndex = 0;
    verifyNextItem();
});

/**
 * Checks the next item in checkoutCart, prompting verification modal if necessary
 */
function verifyNextItem() {
    while (currentItemIndex < checkoutCart.length) {
        const item = checkoutCart[currentItemIndex];
        // If item confidence is less than threshold, prompt for verification
        const confidenceThreshold = 1;
        if (item.confidence < confidenceThreshold) {
            verifyItem(item);
            return;
        }
        currentItemIndex++;
    }
    console.log("Verification complete:", checkoutCart);
    showReviewCartModal();
}

/**
 * Prompts the customer to verify an item
 * @param {*} item item requiring verification
 */
function verifyItem(item) {
    const modal = document.getElementById('verificationModal');
    const itemCanvas = document.getElementById('itemCanvas');
    const ctx = itemCanvas.getContext('2d');
    const itemNameElem = document.getElementById('detectedItem');
    const searchInput = document.getElementById('searchInput');

    modal.style.display = 'block';

    // Set canvas size according to the aspect ratio of item bounding box
    const [x1, y1, x2, y2] = item.bbox;
    const itemWidth = x2 - x1;
    const itemHeight = y2 - y1;
    itemCanvas.width = Math.min(300, itemWidth);
    itemCanvas.height = (itemCanvas.width / itemWidth) * itemHeight;
    
    ctx.clearRect(0, 0, itemCanvas.width, itemCanvas.height);
    ctx.drawImage(detectionImage, x1, y1, itemWidth, itemHeight, 0, 0, itemCanvas.width, itemCanvas.height);

    // Display detected item name
    itemNameElem.textContent = item.display_name;
    
    // Set the search input value to the current item's name
    searchInput.value = item.display_name;

    // Store the current item's index in the modal for reference
    modal.dataset.currentItemIndex = currentItemIndex;

    // Populate suggestions with the top 5 predictions
    populateSuggestions(item.top5_predictions);

    // Search input filtering
    searchInput.oninput = (e) => {
        const searchText = e.target.value.toLowerCase();
        
        // Filter suggestions based on search text
        const filteredSuggestions = product_list
            .filter(product => product.DisplayName.toLowerCase().includes(searchText))
            .map(product => product.DisplayName);

        // Populate suggestions with filtered suggestions or top 5 predictions
        populateSuggestions(filteredSuggestions.length > 0 ? filteredSuggestions : item.top5_predictions);
    };
}

/**
 * Populates the suggestions list with the given suggestions
 * @param {Array} suggestions List of suggestions to populate in the suggestions list
 */
function populateSuggestions(suggestions) {
    const suggestionsList = document.getElementById('suggestionsList');
    suggestionsList.innerHTML = '';

    suggestions.forEach(suggestion => {
        const listItem = document.createElement('li');
        listItem.textContent = suggestion;
        listItem.onclick = () => {
            document.getElementById('searchInput').value = suggestion;
            console.log('Selected item:', suggestion);
        };
        suggestionsList.appendChild(listItem);
    });
}

/**
 * Closes the verification modal
 */
function closeModal() {
    document.getElementById('verificationModal').style.display = 'none';
}

// Function to handle when the user clicks "Continue"
function continueAction() {
    const selectedItemName = document.getElementById('searchInput').value;
    const modal = document.getElementById('verificationModal');
    
    // Get the current item's index from the modal's dataset
    const itemIndex = parseInt(modal.dataset.currentItemIndex);
    
    // Find the selected item in product_list
    if (selectedItemName) {
        const selectedProduct = product_list.find(product => product.DisplayName === selectedItemName);
        console.log('Selected product:', selectedProduct);
        if (selectedProduct) {
            // Create a new object with the updated properties
            const updatedItem = {
                ...checkoutCart[itemIndex],  // Preserve original properties like bbox
                SKU: selectedProduct.SKU,
                barcode: selectedProduct.barcode,
                display_name: selectedProduct.DisplayName,
                item: selectedProduct.ProductDescription,
                lookupcode: selectedProduct.lookupcode,
                price: selectedProduct.price,
                user_input: true
            };
            
            // Update only the specific item in checkoutCart
            checkoutCart[itemIndex] = updatedItem;
            console.log('Updated item:', updatedItem);
        } else {
            console.log("Product not found in product_list");
        }
    }

    closeModal();
    currentItemIndex++; // Move to the next item
    verifyNextItem(); // Continue to next item in checkoutCart that needs verification
}

// Event listeners for cancel and continue buttons
document.addEventListener("DOMContentLoaded", () => {
    const cancelButton = document.querySelector(".verification-close-btn");
    const continueButton = document.querySelector(".verification-continue-btn");

    if (cancelButton) {
        cancelButton.addEventListener("click", closeModal);
    }
    
    if (continueButton) {
        continueButton.addEventListener("click", continueAction);
    }

    // Updated selectors with IDs instead of class names
    const checkoutCloseButton = document.getElementById('checkout-close-btn');
    const checkoutPayButton = document.getElementById('checkout-pay-btn');

    if (checkoutCloseButton) {
        checkoutCloseButton.addEventListener('click', closeReviewModal);
    }

    if (checkoutPayButton) {
        checkoutPayButton.addEventListener('click', confirmPurchase);
    }

    const thankYouCloseButton = document.getElementById('thank-you-close-btn');

    if (thankYouCloseButton) {
        thankYouCloseButton.addEventListener('click', closeThankYouModal);
    }
});

/**
 * Show the Review Cart Modal with a summary of all items and total.
 */
function showReviewCartModal() {
    const reviewModal = document.getElementById('reviewCartModal');
    const cartSummary = document.getElementById('cartSummary');
    const totalAmount = document.getElementById('totalAmount');

    // Clear previous cart summary content
    cartSummary.innerHTML = '';

    // Calculate and display cart items with total amount
    let total = 0;
    checkoutCart.forEach(item => {
        const itemRow = document.createElement('div');
        itemRow.className = 'cart-item';
        
        const nameElem = document.createElement('div');
        nameElem.textContent = item.display_name;
        
        const priceElem = document.createElement('div');
        priceElem.textContent = `$${parseFloat(item.price).toFixed(2)}`;
        
        itemRow.appendChild(nameElem);
        itemRow.appendChild(priceElem);
        cartSummary.appendChild(itemRow);

        total += parseFloat(item.price);
    });

    // Display total amount
    totalAmount.textContent = `$${total.toFixed(2)}`;

    // Show the review modal
    reviewModal.style.display = 'block';
}

/**
 * Close the Review Cart Modal
 */
function closeReviewModal() {
    console.log('Closing Review Cart Modal');
    document.getElementById('reviewCartModal').style.display = 'none';
}

/**
 * Confirm purchase, show the Thank You modal
 */
function confirmPurchase() {
    closeReviewModal();
    showThankYouModal();
    saveData();
}

/**
 * Show the Thank You Modal and set it to close after 3 seconds
 */
function showThankYouModal() {
    const thankYouModal = document.getElementById('thankYouModal');
    thankYouModal.style.display = 'block';

    // Automatically close the thank you modal after 3 seconds
    setTimeout(() => {
        closeThankYouModal();
    }, 3000);
}

/**
 * Close the Thank You Modal
 */
function closeThankYouModal() {
    document.getElementById('thankYouModal').style.display = 'none';
}

/**
 * Sends detection image, customer view image and transaction data to backend for saving
 */
async function saveData() {
    try {
        // Save detection image
        const detectionImageBlob = await new Promise(resolve => {
            detectionImage.toBlob(resolve, 'image/png');
        });
        const itemImageFile = new File([detectionImageBlob], 'item.png', { type: 'image/png' });

        // Save customer view canvas image
        const customerView = document.getElementById('customerView');
        const customerImageBlob = await new Promise(resolve => {
            customerView.toBlob(resolve, 'image/png');
        });
        const customerImageFile = new File([customerImageBlob], 'customer.png', { type: 'image/png' });

        // Create transaction data
        const transactionData = {
            cartId: '1',
            date: new Date().toISOString(),
            items: checkoutCart
        };

        // Create form data to send all files together
        const formData = new FormData();
        formData.append('itemImage', itemImageFile);
        formData.append('customerImage', customerImageFile);
        formData.append('transactionData', new Blob([JSON.stringify(transactionData)], { type: 'application/json' }));

        // Send data to server
        const response = await fetch('http://localhost:8000/save-transaction', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Failed to save transaction data');
        }

        console.log('Transaction data and images saved successfully');

    } catch (error) {
        console.error('Error saving transaction data:', error);
    }
}