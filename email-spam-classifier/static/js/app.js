(() => {
  const spamSample = "CONGRATULATIONS! You've been selected to claim a FREE $1,000 gift card. Click now: www.example.com/claim. Reply STOP to opt out.";
  const hamSample = "Hi Sam, just checking that we're still on for lunch tomorrow at 12:30. I'll bring the project notes. See you then!";

  const showToast = (message, isError = false) => {
    const toast = document.querySelector("[data-toast]");
    if (!toast) return;
    toast.textContent = message;
    toast.classList.toggle("toast-error", isError);
    toast.classList.add("toast-visible");
    window.setTimeout(() => toast.classList.remove("toast-visible"), 3200);
  };

  const formatPercent = (value) => `${(Number(value) * 100).toFixed(1)}%`;

  const escapeText = (value) => {
    const element = document.createElement("span");
    element.textContent = value ?? "";
    return element.innerHTML;
  };

  const loadHealth = async () => {
    try {
      const response = await fetch("/api/health");
      const health = await response.json();
      const ready = health.model_available;
      document.querySelectorAll("[data-model-dot]").forEach((dot) => dot.classList.toggle("is-ready", ready));
      document.querySelectorAll("[data-model-status]").forEach((label) => { label.textContent = ready ? "Model ready" : "Model not trained"; });
      document.querySelectorAll("[data-model-name]").forEach((label) => { label.textContent = ready ? health.model_name : "Train to enable predictions"; });
      const warning = document.querySelector("[data-model-warning]");
      if (warning && !ready) warning.hidden = false;
      const title = document.querySelector("[data-home-model-title]");
      const copy = document.querySelector("[data-home-model-copy]");
      if (title) title.textContent = ready ? health.model_name : "No model loaded yet";
      if (copy) copy.textContent = ready ? "Predictions use a model trained once from labeled SMS data." : "Train the model using the real UCI dataset. The app will not guess or show invented scores.";
    } catch {
      document.querySelectorAll("[data-model-status]").forEach((label) => { label.textContent = "API unavailable"; });
    }
  };

  const bindPrediction = () => {
    const form = document.getElementById("prediction-form");
    if (!form) return;
    const textarea = document.getElementById("message-text");
    const counter = document.querySelector("[data-character-count]");
    const error = document.querySelector("[data-form-error]");
    const submitButton = form.querySelector("button[type='submit']");

    textarea.addEventListener("input", () => { counter.textContent = textarea.value.length.toLocaleString(); });
    document.querySelectorAll("[data-sample]").forEach((button) => {
      button.addEventListener("click", () => {
        textarea.value = button.dataset.sample === "spam" ? spamSample : hamSample;
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
        textarea.focus();
      });
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      error.hidden = true;
      error.textContent = "";
      const text = textarea.value.trim();
      if (!text) {
        error.textContent = "Add a message before running the analysis.";
        error.hidden = false;
        textarea.focus();
        return;
      }
      submitButton.disabled = true;
      submitButton.querySelector(".button-content").hidden = true;
      submitButton.querySelector(".button-loading").hidden = false;
      try {
        const response = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Prediction could not be completed.");
        const container = document.getElementById("prediction-result");
        const placeholder = container.querySelector(".result-placeholder");
        const content = container.querySelector(".result-content");
        const spam = result.prediction === "spam";
        placeholder.hidden = true;
        content.hidden = false;
        container.classList.toggle("result-spam", spam);
        container.classList.toggle("result-ham", !spam);
        const badge = container.querySelector("[data-result-badge]");
        badge.className = `result-badge ${spam ? "badge-spam" : "badge-ham"}`;
        badge.querySelector("i").className = `bi ${spam ? "bi-exclamation-triangle-fill" : "bi-check-circle-fill"}`;
        badge.querySelector("span").textContent = spam ? "SPAM" : "HAM · LEGITIMATE";
        container.querySelector("[data-result-model]").textContent = result.model;
        container.querySelector("[data-confidence-value]").textContent = formatPercent(result.confidence);
        container.querySelector("[data-confidence-bar]").style.width = `${Math.round(result.confidence * 100)}%`;
        container.querySelector("[data-spam-probability]").textContent = formatPercent(result.spam_probability);
        showToast("Message analyzed and saved to local history.");
      } catch (requestError) {
        error.textContent = requestError.message;
        error.hidden = false;
      } finally {
        submitButton.disabled = false;
        submitButton.querySelector(".button-content").hidden = false;
        submitButton.querySelector(".button-loading").hidden = true;
      }
    });
  };

  const bindHistory = async () => {
    const body = document.querySelector("[data-history-body]");
    if (!body) return;
    const empty = document.querySelector("[data-history-empty]");
    const count = document.querySelector("[data-history-count]");
    const renderHistory = async () => {
      const response = await fetch("/api/history?limit=50");
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "History could not be loaded.");
      const items = data.items;
      count.textContent = `${items.length} ${items.length === 1 ? "record" : "records"}`;
      body.replaceChildren();
      empty.hidden = items.length > 0;
      body.closest(".table-responsive").hidden = items.length === 0;
      items.forEach((item) => {
        const row = document.createElement("tr");
        const spam = item.prediction === "spam";
        const date = new Date(item.created_at);
        row.innerHTML = `<td class="preview-cell">${escapeText(item.preview)}</td><td><span class="table-badge ${spam ? "table-badge-spam" : "table-badge-ham"}">${spam ? "<i class=\"bi bi-exclamation-circle-fill\"></i> SPAM" : "<i class=\"bi bi-check-circle-fill\"></i> HAM"}</span></td><td class="confidence-cell">${formatPercent(item.confidence)}</td><td class="muted-cell">${Number(item.character_count).toLocaleString()}</td><td class="muted-cell">${escapeText(date.toLocaleString())}</td>`;
        body.append(row);
      });
    };
    try {
      await renderHistory();
    } catch {
      body.innerHTML = "<tr><td colspan='5' class='history-error'>Could not load history. Check that the local app is running.</td></tr>";
    }
    document.querySelector("[data-clear-history]").addEventListener("click", async (event) => {
      if (!window.confirm("Clear all locally saved prediction history? This cannot be undone.")) return;
      const button = event.currentTarget;
      button.disabled = true;
      try {
        const response = await fetch("/api/history", { method: "DELETE" });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "History could not be cleared.");
        await renderHistory();
        showToast(data.message);
      } catch (requestError) {
        showToast(requestError.message, true);
      } finally {
        button.disabled = false;
      }
    });
  };

  const bindNavigation = () => {
    const toggle = document.querySelector(".menu-toggle");
    const sidebar = document.getElementById("sidebar");
    toggle?.addEventListener("click", () => {
      const isOpen = sidebar.classList.toggle("sidebar-open");
      toggle.setAttribute("aria-expanded", String(isOpen));
    });
    sidebar?.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => {
      sidebar.classList.remove("sidebar-open");
      toggle?.setAttribute("aria-expanded", "false");
    }));
  };

  document.addEventListener("DOMContentLoaded", () => {
    const date = document.querySelector("[data-current-date]");
    if (date) date.textContent = new Intl.DateTimeFormat(undefined, { weekday: "short", month: "short", day: "numeric" }).format(new Date());
    loadHealth();
    bindPrediction();
    bindHistory();
    bindNavigation();
  });
})();