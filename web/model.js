const form = document.querySelector('#translateForm');
const input = document.querySelector('#sourceText');
const button = document.querySelector('#translateButton');
const result = document.querySelector('#result');
const modelInfo = document.querySelector('#modelInfo');

fetch('/api/status').then(r => r.json()).then(data => {
  modelInfo.textContent = data.model_ready
    ? `Modèle chargé · cibles tirées au sort : ${data.targets.map(x => x.name).join(', ')}`
    : 'Poids du modèle absents : préparez les données et entraînez-le.';
}).catch(() => { modelInfo.textContent = 'Application inaccessible.'; });

form.addEventListener('submit', async event => {
  event.preventDefault();
  button.disabled = true;
  result.setAttribute('aria-busy', 'true');
  result.replaceChildren(message('Le modèle prépare une traduction…'));
  try {
    const response = await fetch('/api/translate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: input.value})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'La traduction a échoué.');
    const card = document.createElement('article');
    card.className = 'result-card';
    const lang = document.createElement('p');
    lang.className = 'target-language';
    lang.textContent = `Langue tirée au sort : ${data.target_name}`;
    const originalTitle = document.createElement('h2');
    originalTitle.textContent = 'Texte original';
    const original = document.createElement('p');
    original.className = 'text-block';
    original.textContent = data.source_text;
    const translationTitle = document.createElement('h2');
    translationTitle.textContent = 'Traduction générée par le Transformer';
    const translation = document.createElement('p');
    translation.className = 'text-block';
    translation.textContent = data.translation || 'Le modèle n’a généré aucun texte.';
    card.append(lang, originalTitle, original, translationTitle, translation);
    if (data.pivot_text) {
      const pivot = document.createElement('p');
      pivot.className = 'source';
      pivot.textContent = `Étape intermédiaire en anglais : ${data.pivot_text}`;
      card.append(pivot);
    }
    result.replaceChildren(card);
    button.textContent = 'Nouvelle traduction';
  } catch (error) {
    result.replaceChildren(message(error.message));
  } finally {
    button.disabled = false;
    result.setAttribute('aria-busy', 'false');
  }
});

function message(text) {
  const p = document.createElement('p');
  p.className = 'notice';
  p.textContent = text;
  return p;
}

