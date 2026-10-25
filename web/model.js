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
    const referenceTitle = document.createElement('h2');
    referenceTitle.textContent = 'Correspondances du dictionnaire MUSE';
    const reference = document.createElement('p');
    reference.className = 'text-block';
    renderTokens(reference, data.reference_tokens, 'Aucune entrée correspondante dans le dictionnaire.');
    const translationTitle = document.createElement('h2');
    translationTitle.textContent = 'Prédiction du Transformer, mot par mot (expérimentale)';
    const translation = document.createElement('p');
    translation.className = 'text-block';
    renderTokens(translation, data.model_tokens, 'Aucune prédiction exploitable du modèle.');
    card.append(lang, originalTitle, original, referenceTitle, reference, translationTitle, translation);
    if (data.missing_words?.length) {
      const missing = document.createElement('p');
      missing.className = 'source';
      missing.textContent = `Mots absents du dictionnaire : ${data.missing_words.join(', ')}`;
      card.append(missing);
    }
    if (data.pivot_text) {
      const pivot = document.createElement('p');
      pivot.className = 'source';
      pivot.textContent = `Étape pivot (anglais) : ${data.pivot_text}`;
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

function renderTokens(container, tokens, emptyMessage) {
  if (!tokens?.length) {
    container.textContent = emptyMessage;
    return;
  }
  tokens.forEach((item, index) => {
    if (index) container.append(document.createTextNode(' '));
    const span = document.createElement('span');
    span.className = item.available ? 'translated-token' : 'missing-token';
    span.textContent = item.translation || `[${item.source} : indisponible]`;
    container.append(span);
  });
}
