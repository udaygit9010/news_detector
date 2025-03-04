function checkNews() {
    let newsText = document.getElementById("newsInput").value;
    let resultDiv = document.getElementById("result");

    if (newsText.trim() === "") {
        resultDiv.innerHTML = "⚠️ Please enter some text!";
        resultDiv.style.color = "#ffc107";
        resultDiv.style.display = "block";
        return;
    }

    resultDiv.innerHTML = "⏳ Checking...";
    resultDiv.style.color = "#ffffff";
    resultDiv.style.display = "block";

    fetch('/predict', {
        method: 'POST',
        body: new URLSearchParams({ "news_text": newsText }),
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    })
    .then(response => response.json())
    .then(data => {
        resultDiv.innerHTML = data.result;
        resultDiv.style.color = data.color;
    })
    .catch(error => {
        resultDiv.innerHTML = "❌ Error: Unable to process.";
        resultDiv.style.color = "#dc3545";
    });
}
