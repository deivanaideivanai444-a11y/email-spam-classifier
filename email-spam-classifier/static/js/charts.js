(() => {
  const metricNames = ["accuracy", "precision", "recall", "f1_score"];
  const metricLabels = { accuracy: "Accuracy", precision: "Precision", recall: "Recall", f1_score: "Spam F1" };
  const colors = ["#6075dd", "#22a88a", "#ed7183", "#efad58"];
  const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);

  const renderComparison = (data) => {
    const summary = document.querySelector("[data-metric-summary]");
    summary.hidden = false;
    metricNames.forEach((key) => {
      document.querySelector(`[data-metric="${key}"]`).textContent = `${(data.best[key] * 100).toFixed(1)}%`;
    });
    document.querySelector("[data-best-model]").textContent = `Selected: ${data.best_model}`;
    document.querySelector("[data-test-rows]").textContent = Number(data.test_rows).toLocaleString();
    document.querySelector("[data-dataset-name]").textContent = data.dataset_name;
    document.querySelector("[data-method-copy]").textContent = `${data.training_rows.toLocaleString()} training and ${data.test_rows.toLocaleString()} held-out messages. Selection uses spam-class F1, with accuracy as the tie-breaker.`;

    const chartEmpty = document.querySelector("[data-metrics-chart-empty]");
    if (window.Chart) {
      new Chart(document.getElementById("metrics-chart"), {
        type: "bar",
        data: { labels: Object.keys(data.models), datasets: metricNames.map((key, index) => ({ label: metricLabels[key], data: Object.values(data.models).map((result) => result[key]), backgroundColor: colors[index], borderRadius: 4, maxBarThickness: 28 })) },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { position: "bottom", labels: { usePointStyle: true, pointStyle: "circle", padding: 18, color: "#687187", font: { family: "DM Sans" } } }, tooltip: { backgroundColor: "#151c33", padding: 12, cornerRadius: 8, callbacks: { label: (context) => `${context.dataset.label}: ${(context.raw * 100).toFixed(1)}%` } } },
          scales: { x: { grid: { display: false }, ticks: { color: "#687187", font: { family: "DM Sans", size: 11 } } }, y: { min: 0, max: 1, ticks: { stepSize: 0.2, callback: (value) => `${value * 100}%`, color: "#858da0" }, grid: { color: "#edf0f6" }, border: { display: false } } },
        },
      });
    } else {
      chartEmpty.hidden = false;
    }

    const matrix = data.best.confusion_matrix;
    const matrixContainer = document.querySelector("[data-confusion-matrix]");
    matrixContainer.innerHTML = `<table class="confusion-table"><thead><tr><th></th><th>Predicted ham</th><th>Predicted spam</th></tr></thead><tbody><tr><th>Actual ham</th><td class="matrix-correct">${matrix[0][0]}</td><td class="matrix-missed">${matrix[0][1]}</td></tr><tr><th>Actual spam</th><td class="matrix-missed">${matrix[1][0]}</td><td class="matrix-correct">${matrix[1][1]}</td></tr></tbody></table>`;

    ["spam", "ham"].forEach((group) => {
      const container = document.querySelector(`[data-features="${group}"]`);
      const features = data.important_features[group] || [];
      container.innerHTML = features.length
        ? features.map((item) => `<span class="feature-chip ${group === "spam" ? "feature-spam" : "feature-ham"}">${escapeHtml(item.word)}</span>`).join("")
        : "<span class='no-features'>No linear term weights are available for this model.</span>";
    });
  };

  document.addEventListener("DOMContentLoaded", async () => {
    if (!document.querySelector("[data-metric-summary]")) return;
    try {
      const response = await fetch("/api/metrics");
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Model evaluation is unavailable.");
      renderComparison(data);
    } catch (error) {
      const warning = document.querySelector("[data-metrics-warning]");
      warning.hidden = false;
      document.querySelector("[data-metrics-error]").textContent = error.message;
      document.querySelector("[data-metrics-chart-empty]").hidden = false;
    }
  });
})();