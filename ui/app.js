const apiBase = "http://127.0.0.1:8000";
const evaluateBtn = document.getElementById("evaluateBtn");
const tripIdInput = document.getElementById("tripId");
const predictionMethod = document.getElementById("predictionMethod");
const statusText = document.getElementById("status");
const map = document.getElementById("map");
const scoreIds = ["accuracy", "precision", "recall", "f1"];
const metricIds = [...scoreIds, "observationCount", "candidateMisses", "anomalies", "processingTime"];

function setMetric(id, value) {
    document.getElementById(id).textContent = value ?? "—";
}

function drawRoute(result) {
    const track = result.track ?? [];
    const segments = Object.values(result.geometry ?? {});
    if (!track.length) {
        map.textContent = "No GPS geometry was returned. Restart the backend with the updated API.";
        return;
    }

    const points = [...track, ...segments.flatMap(segment => [segment.start, segment.end])];
    const longitudeScale = Math.cos(track[0].lat * Math.PI / 180);
    const bounds = points.reduce((box, point) => ({
        west: Math.min(box.west, point.lon),
        east: Math.max(box.east, point.lon),
        south: Math.min(box.south, point.lat),
        north: Math.max(box.north, point.lat)
    }), { west: Infinity, east: -Infinity, south: Infinity, north: -Infinity });
    const width = (bounds.east - bounds.west) * longitudeScale;
    const height = bounds.north - bounds.south;
    const scale = Math.min(760 / (width || 1), 340 / (height || 1));
    const position = point => [
        400 + ((point.lon - bounds.west) * longitudeScale - width / 2) * scale,
        190 + (height / 2 - (point.lat - bounds.south)) * scale
    ].map(value => value.toFixed(2)).join(",");
    const svgNode = (tag, attributes) => {
        const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
        for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
        return node;
    };
    const svg = svgNode("svg", {
        viewBox: "0 0 800 400", width: "100%", height: "100%",
        role: "img", "aria-label": "GPS trace in gray and predicted road segments in blue"
    });
    const gps = svgNode("g", { fill: "#64748b", opacity: "0.55" });
    for (const point of track) {
        const [cx, cy] = position(point).split(",");
        gps.append(svgNode("circle", { cx, cy, r: "1.5" }));
    }
    svg.append(gps);
    svg.append(svgNode("path", {
        d: segments.map(segment => `M${position(segment.start)}L${position(segment.end)}`).join(" "),
        fill: "none", stroke: "#2563eb", "stroke-width": "2"
    }));
    const legend = svgNode("text", { x: "20", y: "390", fill: "#334155", "font-size": "14" });
    legend.textContent = segments.length ? "Gray: GPS fixes · Blue: predicted road segments" : "Gray: GPS fixes · No predicted segments";
    svg.append(legend);
    map.replaceChildren(svg);
}

tripIdInput.type = "text";
tripIdInput.inputMode = "numeric";
predictionMethod.addEventListener("change", () => {
    if (!evaluateBtn.disabled) statusText.textContent = "Click Evaluate Trip to run the selected method.";
});

evaluateBtn.addEventListener("click", async () => {
    const input = tripIdInput.value.trim();
    if (!/^\d{1,8}$/.test(input)) {
        statusText.textContent = "Enter a numeric trip ID of up to eight digits.";
        return;
    }

    const tripId = input.padStart(8, "0");
    tripIdInput.value = tripId;
    const useViterbi = predictionMethod.value === "true";
    const method = useViterbi ? "Viterbi" : "Greedy";
    metricIds.forEach(id => setMetric(id));
    map.textContent = "Loading route…";
    statusText.textContent = `Running ${method} for trip ${tripId}…`;
    evaluateBtn.disabled = true;
    predictionMethod.disabled = true;

    try {
        const response = await fetch(`${apiBase}/predict/${tripId}?use_viterbi=${useViterbi}`, {
            method: "POST"
        });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(typeof result.detail === "string" ? result.detail : `HTTP ${response.status}`);
        }

        for (const id of scoreIds) {
            setMetric(id, typeof result[id] === "number" ? `${(result[id] * 100).toFixed(2)}%` : "Unavailable");
        }
        setMetric("observationCount", result.observation_count);
        setMetric("candidateMisses", result.candidate_miss_count);
        setMetric("anomalies", result.anomalous_observations_count);
        setMetric("processingTime", typeof result.processing_time_ms === "number" ? `${result.processing_time_ms.toFixed(2)} ms` : null);
        drawRoute(result);
        const scoreStatus = result.evaluated_observation_count
            ? ` Scores use ${result.evaluated_observation_count} aligned fixes; precision, recall, and F1 are macro-averaged.`
            : " No aligned reference labels were returned for scoring.";
        const model = result.model_name ? `${result.model_name} + ` : "";
        statusText.textContent = `Trip ${tripId} completed using ${model}${method}.${scoreStatus}`;
    } catch (error) {
        map.textContent = "Route unavailable.";
        statusText.textContent = `Could not evaluate trip ${tripId}: ${error.message}`;
    } finally {
        evaluateBtn.disabled = false;
        predictionMethod.disabled = false;
    }
});
