const qs = (s, el=document) => el.querySelector(s);
const qsa = (s, el=document) => [...el.querySelectorAll(s)];

async function askAgent(prompt) {
  const res = await fetch('/api/ask', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt })
  });
  const data = await res.json();
  if (res.ok) return data.answer || '';
  throw new Error(data.error || 'ask failed');
}

function streamImages({ prompt, answer, topic, delay=0.25, onItem, onEnd }) {
  const params = new URLSearchParams({ prompt, delay: String(delay) });
  if (answer) params.set('answer', answer);
  if (topic) params.set('topic', topic);
  const es = new EventSource('/api/images/stream?' + params.toString());
  es.onmessage = (ev) => {
    if (!ev.data) return;
    try {
      const data = JSON.parse(ev.data);
      if (data.url && onItem) onItem(data);
      if (data.info === 'no-images') {
        onEnd && onEnd('No related images found. Try a clearer topic.');
        es.close();
      }
    } catch {}
  };
  es.onerror = () => { es.close(); onEnd && onEnd('image stream ended'); };
  return () => es.close();
}

function addImageCard({ url, title, page_url, width, height }) {
  const grid = qs('.images');
  const fig = document.createElement('figure');
  fig.className = 'figure fade-in';
  // Image link (opens full image)
  const aImg = document.createElement('a');
  aImg.href = url; aImg.target = '_blank'; aImg.rel = 'noopener noreferrer'; aImg.title = url;
  const img = document.createElement('img');
  img.src = url; img.alt = title || '';
  if (width && height) {
    // only set aspect ratio, not width/height attributes
    img.style.aspectRatio = `${width} / ${height}`;
  }
  aImg.appendChild(img);
  const cap = document.createElement('figcaption');
  cap.textContent = title || '';
  // Title link (opens article page)
  if (page_url) {
    const link = document.createElement('a');
    link.href = page_url; link.target = '_blank'; link.rel = 'noopener noreferrer';
    link.textContent = title || 'View source';
    cap.textContent = '';
    cap.appendChild(link);
  }
  fig.appendChild(aImg); fig.appendChild(cap);
  grid.prepend(fig);
}

async function main() {
  const form = qs('#form');
  const promptEl = qs('#prompt');
  const topicEl = qs('#topic');
  const delayEl = qs('#delay');
  const answerEl = qs('#answer');
  const imagesGrid = qs('.images');

  let cancelStream = null;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const prompt = promptEl.value.trim();
    const topic = topicEl.value.trim();
    const delay = Number(delayEl.value || 0.25);

    imagesGrid.innerHTML = '';
    answerEl.textContent = 'Thinking…';

    try {
      const answer = await askAgent(prompt);
      answerEl.textContent = answer;
      cancelStream && cancelStream();
  cancelStream = streamImages({ prompt, answer, topic, delay,
        onItem: addImageCard,
        onEnd: (msg) => { if (msg) console.log(msg); }
      });
    } catch (e) {
      answerEl.textContent = 'Error: ' + (e?.message || String(e));
    }
  });
}

window.addEventListener('DOMContentLoaded', main);
