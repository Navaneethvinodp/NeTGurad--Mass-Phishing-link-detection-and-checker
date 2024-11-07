document.addEventListener('DOMContentLoaded', function() {
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    let currentUrl = tabs[0].url;
    document.getElementById('url').textContent = currentUrl;

    chrome.runtime.sendMessage({action: "getSafetyInfo", url: currentUrl}, function(response) {
      document.getElementById('safety-score').textContent = response.safety_score.toFixed(2);
      let statusElement = document.getElementById('status');
      
      if (response.is_safe) {
        statusElement.textContent = "Safe";
        statusElement.className = "safe";
      } else {
        statusElement.textContent = "Caution";
        statusElement.className = "warning";
      }

      // Display detailed information
      displayDetailedInfo(response.details);
    });
  });
});

function displayDetailedInfo(details) {
  let detailsElement = document.getElementById('detailed-info');
  detailsElement.innerHTML = `
    <h3>Detailed Information:</h3>
    <p>Google Safe Browsing: ${details.google_safe_browsing.safe ? 'Safe' : 'Unsafe'}</p>
    <p>SSL Grade: ${details.ssl_info.grade}</p>
    <p>Domain Age: ${details.domain_info.age.toFixed(0)} days</p>
    <p>Suspicious Word Count: ${details.content_analysis.suspiciousWordCount}</p>
    <p>External Link Count: ${details.content_analysis.externalLinkCount}</p>
  `;
}


