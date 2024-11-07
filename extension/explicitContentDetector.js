function detectExplicitContent() {
  const explicitWords = ['xxx', 'porn', 'adult', 'explicit', 'nsfw'];
  const bodyText = document.body.innerText.toLowerCase();
  const images = document.getElementsByTagName('img');

  let isExplicit = explicitWords.some(word => bodyText.includes(word));

  if (!isExplicit) {
    // Check image alt texts and src URLs for explicit content
    for (let img of images) {
      const altText = img.alt.toLowerCase();
      const srcUrl = img.src.toLowerCase();
      if (explicitWords.some(word => altText.includes(word) || srcUrl.includes(word))) {
        isExplicit = true;
        break;
      }
    }
  }

  return isExplicit;
}

function blurExplicitImages() {
  const images = document.getElementsByTagName('img');
  for (let img of images) {
    img.classList.add('blurred');
  }
}

window.detectExplicitContent = detectExplicitContent;
window.blurExplicitImages = blurExplicitImages;