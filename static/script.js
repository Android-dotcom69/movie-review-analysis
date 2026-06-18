const textarea    = document.getElementById('review-input');
const analyzeBtn  = document.getElementById('analyze-btn');
const btnText     = analyzeBtn.querySelector('.btn-text');
const btnSpinner  = analyzeBtn.querySelector('.btn-spinner');
const charCount   = document.getElementById('char-count');
const clearBtn    = document.getElementById('clear-btn');

const resultCard  = document.getElementById('result-card');
const resultIcon  = document.getElementById('result-icon');
const resultLabel = document.getElementById('result-label');
const confValue   = document.getElementById('confidence-value');
const confBar     = document.getElementById('confidence-bar');
const resultNote  = document.getElementById('result-note');

const errorCard   = document.getElementById('error-card');
const errorMsg    = document.getElementById('error-msg');

const POSITIVE_NOTES = [
  "The model detected an upbeat, appreciative tone.",
  "Strong positive language and enthusiasm detected.",
  "The review reads as a recommendation.",
];
const NEGATIVE_NOTES = [
  "The model picked up on critical or disappointed language.",
  "Negative sentiment and dissatisfaction detected.",
  "The review leans towards discouraging the film.",
];

textarea.addEventListener('input', () => {
  const len = textarea.value.length;
  charCount.textContent = `${len} / 5000`;
  hideResults();
});

clearBtn.addEventListener('click', () => {
  textarea.value = '';
  charCount.textContent = '0 / 5000';
  hideResults();
  textarea.focus();
});

document.querySelectorAll('.example-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    textarea.value = chip.dataset.text;
    charCount.textContent = `${chip.dataset.text.length} / 5000`;
    hideResults();
    textarea.focus();
  });
});

analyzeBtn.addEventListener('click', async () => {
  const review = textarea.value.trim();
  if (!review) {
    showError('Please enter a review before analyzing.');
    return;
  }

  setLoading(true);
  hideResults();

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ review }),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.error || 'Something went wrong. Please try again.');
      return;
    }

    showResult(data);
  } catch (err) {
    showError('Could not reach the server. Is it running?');
  } finally {
    setLoading(false);
  }
});

function showResult({ sentiment, confidence }) {
  const isPositive = sentiment === 'positive';

  resultCard.hidden = false;
  resultCard.className = `result-card ${isPositive ? 'is-positive' : 'is-negative'}`;
  resultIcon.textContent  = isPositive ? '🎉' : '😞';
  resultLabel.textContent = isPositive ? 'Positive' : 'Negative';
  confValue.textContent   = `${confidence}%`;

  const notes = isPositive ? POSITIVE_NOTES : NEGATIVE_NOTES;
  resultNote.textContent  = notes[Math.floor(Math.random() * notes.length)];

  errorCard.hidden = true;

  // Animate bar after paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      confBar.style.width = `${confidence}%`;
    });
  });
}

function showError(msg) {
  errorCard.hidden = false;
  errorMsg.textContent = msg;
  resultCard.hidden = true;
}

function hideResults() {
  resultCard.hidden = true;
  errorCard.hidden  = true;
  confBar.style.width = '0%';
}

function setLoading(loading) {
  analyzeBtn.disabled = loading;
  btnText.textContent = loading ? 'Analyzing…' : 'Analyze Sentiment';
  btnSpinner.hidden   = !loading;
}
