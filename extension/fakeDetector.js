// Basic checks for fake website indicators
function checkForFakeWebsiteIndicators() {
  const indicators = {
    suspiciousDomain: checkSuspiciousDomain(),
    poorSSL: checkSSLCertificate(),
    suspiciousContent: checkPageContent(),
    abnormalStructure: checkPageStructure()
  };

  const score = calculateFakeScore(indicators);
  return { indicators, score };
}

function checkSuspiciousDomain() {
  const domain = window.location.hostname;
  // Check for suspicious TLDs, excessive subdomains, etc.
  return domain.split('.').length > 3 || /\d{3,}/.test(domain);
}

function checkSSLCertificate() {
  // Basic check - in real implementation, this would be more thorough
  return !window.location.protocol.includes('https');
}

function checkPageContent() {
  const bodyText = document.body.innerText.toLowerCase();
  const suspiciousWords = ['free', 'win', 'congratulation', 'urgent', 'limited time'];
  return suspiciousWords.some(word => bodyText.includes(word));
}

function checkPageStructure() {
  // Check for suspicious structures like too few elements, hidden elements, etc.
  return document.body.children.length < 5;
}

function calculateFakeScore(indicators) {
  // Simple scoring system - can be made more sophisticated
  let score = 0;
  for (let key in indicators) {
    if (indicators[key]) score++;
  }
  return (score / Object.keys(indicators).length) * 100;
}

// Export the main function
window.checkForFakeWebsiteIndicators = checkForFakeWebsiteIndicators;