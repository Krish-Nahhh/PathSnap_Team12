
const evaluateBtn = document.getElementById("evaluateBtn");
const tripIdInput = document.getElementById("tripId");
const predictionMethod = document.getElementById("predictionMethod");
const statusText = document.getElementById("status");

evaluateBtn.addEventListener("click", function () {

    const tripId = tripIdInput.value.trim();
    const useViterbi = predictionMethod.value;

    if (tripId === "") {
        statusText.textContent = "Please enter a Trip ID.";
        return;
    }

    if (Number(tripId) < 0) {
        statusText.textContent = "Please enter a valid Trip ID.";
        return;
    }

    const methodName = useViterbi === "true"
        ? "Viterbi"
        : "Greedy";

    statusText.textContent =
        `Trip ${tripId} selected. Prediction method: ${methodName}. Waiting for backend connection...`;

});
