document.getElementById('summarizeBtn').addEventListener('click', async () => {
    const videoId = document.getElementById('videoId').value;
    const summaryResult = document.getElementById('summaryResult');
    const spinner = document.getElementById('spinner');

    // Hide the summary result initially and show the spinner
    summaryResult.style.display = 'none';
    spinner.style.display = 'block'; // Show the spinner
    summaryResult.textContent = 'Generating summary...';

    try {
        const response = await fetch('http://127.0.0.1:5000/summarize', { // Change the URL if needed
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ video_id: videoId }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Something went wrong');
        }

        // Set summary text and display it
        summaryResult.textContent = data.summary;
        summaryResult.style.display = 'block'; // Show the summary result

    } catch (error) {
        summaryResult.textContent = `Error: ${error.message}`;
        summaryResult.style.display = 'block'; // Show error message
    } finally {
        spinner.style.display = 'none'; // Hide the spinner when done
    }
});
