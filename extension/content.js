// Listen for messages from the background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "updateSafetyIcon") {
    updateSafetyDisplay(request.data);
  } else if (request.action === "checkExplicitContent") {
    const isExplicit = detectExplicitContent();
    if (isExplicit) {
      blurExplicitImages();
    }
    sendResponse({ isExplicit });
  }
});

// Function to update the safety display
function updateSafetyDisplay(safetyInfo) {
  // Create or update safety indicator
  let safetyIndicator = document.getElementById('safety-indicator');
  if (!safetyIndicator) {
    safetyIndicator = document.createElement('div');
    safetyIndicator.id = 'safety-indicator';
    safetyIndicator.style.position = 'fixed';
    safetyIndicator.style.top = '10px';
    safetyIndicator.style.right = '10px';
    safetyIndicator.style.zIndex = '9999';
    document.body.appendChild(safetyIndicator);
  }

  // Update indicator based on safety info
  safetyIndicator.textContent = safetyInfo.is_safe ? '✅ Safe' : '⚠️ Caution';
  safetyIndicator.style.backgroundColor = safetyInfo.is_safe ? 'green' : 'yellow';
  safetyIndicator.style.color = safetyInfo.is_safe ? 'white' : 'black';
  safetyIndicator.style.padding = '5px';
  safetyIndicator.style.borderRadius = '5px';
}

// Run fake website check
const fakeWebsiteCheck = checkForFakeWebsiteIndicators();

// Check for explicit content
const isExplicit = detectExplicitContent();
if (isExplicit) {
  blurExplicitImages();
}

// Send results to background script
chrome.runtime.sendMessage({
  action: "fakeWebsiteCheck",
  data: { ...fakeWebsiteCheck, isExplicit }
});