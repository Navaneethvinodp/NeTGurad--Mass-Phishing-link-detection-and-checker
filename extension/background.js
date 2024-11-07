const safetyCache = {}; // Cache to store safety information for websites

async function checkWebsiteSafety(url) {
  // Check if the safety information is already cached
  if (safetyCache[url]) {
    return safetyCache[url];
  }

  // In a real implementation, this would make an API call to your backend
  // For now, we'll use a mock response
  const safetyInfo = await new Promise((resolve) => {
    setTimeout(() => {
      const info = {
        safety_score: Math.random() * 20 + 80, // Random score between 80 and 100
        is_safe: Math.random() > 0.5
      };
      // Cache the result
      safetyCache[url] = info;
      resolve(info);
    }, 500);
  });

  return safetyInfo;
}

async function analyzeFakeWebsite(url, fakeCheckData) {
  // This would typically involve sending data to your backend for analysis
  // For now, we'll use the local check results
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        is_fake: fakeCheckData.score > 50,
        confidence: fakeCheckData.score,
        indicators: fakeCheckData.indicators
      });
    }, 500);
  });
}

function redirectToGoogle(tabId) {
  chrome.tabs.update(tabId, { url: "https://www.google.com" });
}

// Event listener for tab updates
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    checkWebsiteSafety(tab.url).then(safetyInfo => {
      chrome.tabs.sendMessage(tabId, {
        action: "updateSafetyIcon",
        data: safetyInfo
      });
    });

    // Check for explicit content
    chrome.tabs.sendMessage(tabId, { action: "checkExplicitContent" }, response => {
      if (response && response.isExplicit) {
        redirectToGoogle(tabId);
      }
    });
  }
});

// Message listener for communication between background script and content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "getSafetyInfo") {
    checkWebsiteSafety(request.url).then(safetyInfo => {
      sendResponse(safetyInfo);
    });
    return true; // Indicates that the response will be sent asynchronously
  } else if (request.action === "fakeWebsiteCheck") {
    analyzeFakeWebsite(sender.tab.url, request.data).then(fakeAnalysis => {
      // Update the safety info with fake website analysis
      chrome.tabs.sendMessage(sender.tab.id, {
        action: "updateSafetyIcon",
        data: { ...fakeAnalysis, is_safe: !fakeAnalysis.is_fake && !request.data.isExplicit }
      });

      // Redirect to Google if explicit content is detected
      if (request.data.isExplicit) {
        redirectToGoogle(sender.tab.id);
      }
    });
  }
});
