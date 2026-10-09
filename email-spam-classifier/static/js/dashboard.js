(() => {
  const renderStats = (stats) => {
    document.querySelectorAll("[data-stat]").forEach((element) => {
      const field = element.dataset.stat;
      const value = field === "total" ? stats.total_predictions
        : field === "spam" ? stats.spam_count
          : field === "ham" ? stats.ham_count : stats.spam_percentage;
      element.textContent = Number(value).toLocaleString();
    });
  };

  const chartColors = { spam: "#ef7183", ham: "#3ebc98", grid: "#edf0f6", text: "#788198" };

  const createActivityChart = (stats) => {
    const canvas = document.getElementById("prediction-chart");
    if (!canvas) return;
    const empty = document.querySelector("[data-prediction-empty]");
    if (!stats.daily.length) {
      empty.hidden = false;
      canvas.hidden = true;
      return;
    }
    empty.hidden = true;
    canvas.hidden = false;
    if (!window.Chart) return;
    const labels = stats.daily.map((item) => new Date(`${item.day}T00:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric" }));
    new Chart(canvas, {
      type: "bar",
      data: { labels, datasets: [
        { label: "Spam", data: stats.daily.map((item) => item.spam), backgroundColor: chartColors.spam, borderRadius: 5, maxBarThickness: 24 },
        { label: "Ham", data: stats.daily.map((item) => item.ham), backgroundColor: chartColors.ham, borderRadius: 5, maxBarThickness: 24 },
      ] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { backgroundColor: "#151c33", padding: 12, cornerRadius: 8 } },
        scales: {
          x: { grid: { display: false }, ticks: { color: chartColors.text, font: { family: "DM Sans" } } },
          y: { beginAtZero: true, ticks: { precision: 0, color: chartColors.text }, grid: { color: chartColors.grid }, border: { display: false } },
        },
      },
    });
  };

  const createMixChart = (stats) => {
    const canvas = document.getElementById("mix-chart");
    if (!canvas || !window.Chart || !stats.total_predictions) return;
    new Chart(canvas, {
      type: "doughnut",
      data: { labels: ["Spam", "Ham"], datasets: [{ data: [stats.spam_count, stats.ham_count], backgroundColor: [chartColors.spam, chartColors.ham], borderWidth: 0, hoverOffset: 5 }] },
      options: { cutout: "76%", responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { backgroundColor: "#151c33", padding: 12, cornerRadius: 8 } } },
    });
  };

  document.addEventListener("DOMContentLoaded", async () => {
    if (!document.querySelector("[data-stats-view]")) return;
    try {
      const response = await fetch("/api/stats");
      const stats = await response.json();
      if (!response.ok) throw new Error(stats.error || "Statistics could not be loaded.");
      renderStats(stats);
      createActivityChart(stats);
      createMixChart(stats);
    } catch {
      document.querySelectorAll("[data-stat]").forEach((element) => { element.textContent = "—"; });
      const empty = document.querySelector("[data-prediction-empty]");
      if (empty) { empty.textContent = "Could not load local activity."; empty.hidden = false; }
    }
  });
})();