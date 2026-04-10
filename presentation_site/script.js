const data = window.presentationData || null;

function renderMetricCards() {
  if (!data || !Array.isArray(data.heroMetrics)) return;
  const metricGrid = document.getElementById("metric-grid");
  if (!metricGrid) return;

  metricGrid.innerHTML = "";
  data.heroMetrics.forEach((metric) => {
    const card = document.createElement("article");
    card.className = "metric-card";
    card.innerHTML = `
      <p class="metric-card-label">${metric.label}</p>
      <p class="metric-card-value">${metric.value}</p>
      <p class="metric-card-detail">${metric.detail}</p>
    `;
    metricGrid.appendChild(card);
  });
}

function renderChallengeFacts() {
  if (!data || !Array.isArray(data.challengeFacts)) return;
  const list = document.getElementById("challenge-facts");
  if (!list) return;
  list.innerHTML = "";
  data.challengeFacts.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}

function renderBlendChips() {
  if (!data || !Array.isArray(data.blendChips)) return;
  const row = document.getElementById("blend-chips");
  if (!row) return;
  row.innerHTML = "";
  data.blendChips.forEach((item) => {
    const span = document.createElement("span");
    span.className = "chip";
    span.textContent = item;
    row.appendChild(span);
  });
}

function renderFeatureNotes() {
  if (!data || !Array.isArray(data.featureNotes)) return;
  const list = document.getElementById("feature-notes");
  if (!list) return;
  list.innerHTML = "";
  data.featureNotes.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}

function renderOutlookMetrics() {
  if (!data || !Array.isArray(data.outlookMetrics)) return;
  const grid = document.getElementById("outlook-metrics");
  if (!grid) return;
  grid.innerHTML = "";
  data.outlookMetrics.forEach((metric) => {
    const card = document.createElement("article");
    card.className = "metric-card";
    card.innerHTML = `
      <p class="metric-card-label">${metric.label}</p>
      <p class="metric-card-value">${metric.value}</p>
      <p class="metric-card-detail">${metric.detail}</p>
    `;
    grid.appendChild(card);
  });
}

function renderOutlookList(targetId, items) {
  if (!Array.isArray(items)) return;
  const list = document.getElementById(targetId);
  if (!list) return;
  list.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}

function renderExperimentTable() {
  if (!data || !Array.isArray(data.experimentRows)) return;
  const body = document.getElementById("experiment-table-body");
  if (!body) return;
  body.innerHTML = "";
  data.experimentRows.forEach((row) => {
    const tr = document.createElement("tr");
    if (row.highlight) {
      tr.classList.add("is-highlight");
    }
    tr.innerHTML = `
      <td>${row.label}</td>
      <td>${row.composite.toFixed(2)}</td>
      <td>${row.maeAll.toFixed(2)}</td>
      <td>${row.maePeak.toFixed(2)}</td>
      <td>${row.pinball.toFixed(2)}</td>
    `;
    body.appendChild(tr);
  });
}

function renderTakeaways() {
  if (!data || !Array.isArray(data.takeaways)) return;
  const container = document.getElementById("takeaways");
  if (!container) return;
  container.innerHTML = "";
  data.takeaways.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "takeaway-card";
    card.innerHTML = `
      <h3>0${index + 1}</h3>
      <p>${item}</p>
    `;
    container.appendChild(card);
  });
}

function renderGeneratedAt() {
  const node = document.getElementById("generated-at");
  if (!node || !data || !data.generatedAt) return;
  node.textContent = `Generated ${data.generatedAt}.`;
}

function setupRevealAnimations() {
  const items = document.querySelectorAll(".reveal");
  if (!items.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.18 }
  );

  items.forEach((item) => observer.observe(item));
}

function setupActiveNav() {
  const links = Array.from(document.querySelectorAll(".site-nav a"));
  const sections = links
    .map((link) => {
      const target = document.querySelector(link.getAttribute("href"));
      return target ? { link, target } : null;
    })
    .filter(Boolean);

  if (!sections.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        sections.forEach(({ link }) => link.classList.remove("is-active"));
        const active = sections.find(({ target }) => target === entry.target);
        if (active) {
          active.link.classList.add("is-active");
        }
      });
    },
    {
      rootMargin: "-35% 0px -45% 0px",
      threshold: 0.01,
    }
  );

  sections.forEach(({ target }) => observer.observe(target));
}

function init() {
  renderMetricCards();
  renderChallengeFacts();
  renderBlendChips();
  renderFeatureNotes();
  renderOutlookMetrics();
  renderOutlookList("outlook-points", data?.outlookPoints);
  renderOutlookList("outlook-steps", data?.outlookSteps);
  renderExperimentTable();
  renderTakeaways();
  renderGeneratedAt();
  setupRevealAnimations();
  setupActiveNav();
}

document.addEventListener("DOMContentLoaded", init);
