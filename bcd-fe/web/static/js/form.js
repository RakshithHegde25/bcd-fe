document.addEventListener('DOMContentLoaded', function () {
    // JSON form submission with better error handling
    const jsonForm = document.getElementById('json-form');
    if (jsonForm) {
        jsonForm.addEventListener('submit', function (e) {
            e.preventDefault();

            const jsonInput = document.getElementById('json-data');
            const resultContainer = document.getElementById('json-result');
            const submitBtn = this.querySelector('button[type="submit"]');

            // Clear previous results/errors
            resultContainer.innerHTML = '';

            // Show loading state
            const originalText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            submitBtn.disabled = true;

            try {
                // First validate JSON syntax
                const jsonData = JSON.parse(jsonInput.value);

                // Then validate content structure
                if (typeof jsonData !== 'object' || jsonData === null) {
                    throw new Error('Input must be a JSON object');
                }

                // Check for required fields (example checks)
                const requiredFields = ['radius_mean', 'texture_mean', 'perimeter_mean'];
                const missingFields = requiredFields.filter(field => !(field in jsonData));

                if (missingFields.length > 0) {
                    throw new Error(`Missing required fields: ${missingFields.join(', ')}`);
                }

                // Submit to backend
                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(jsonData)
                })
                    .then(response => {
                        if (!response.ok) {
                            return response.json().then(err => {
                                throw new Error(err.error || 'Server error');
                            });
                        }
                        return response.json();
                    })
                    .then(data => {
                        showResult(data, resultContainer);
                    })
                    .catch(error => {
                        showError(error.message, resultContainer);
                    })
                    .finally(() => {
                        submitBtn.innerHTML = originalText;
                        submitBtn.disabled = false;
                    });

            } catch (error) {
                showError(error.message, resultContainer);
                submitBtn.innerHTML = originalText;
                submitBtn.disabled = false;
            }
        });
    }

    // Helper function to display results
    function showResult(data, container) {
        container.innerHTML = `
            <div class="result-card">
                <div class="diagnosis-result ${data.diagnosis === 'Malignant' ? 'diagnosis-malignant' : 'diagnosis-benign'}">
                    Diagnosis: ${data.diagnosis}
                </div>
                <div class="result-details">
                    <p><strong>Confidence:</strong> ${data.confidence}%</p>
                    <p><strong>Model Accuracy:</strong> ${data.accuracy}%</p>
                </div>
                <div class="features-toggle" onclick="toggleFeatures()">
                    <i class="fas fa-chevron-down"></i> Show Features Used
                </div>
                <div class="features-container" style="display: none;">
                    <pre>${JSON.stringify(data.features_used, null, 2)}</pre>
                </div>
            </div>
        `;
    }

    // Helper function to display errors
    function showError(message, container) {
        container.innerHTML = `
            <div class="error-card">
                <div class="error-header">
                    <i class="fas fa-exclamation-circle"></i>
                    <span>Invalid Input</span>
                </div>
                <div class="error-message">
                    ${message}
                </div>
                ${message.includes('JSON') ? '<div class="error-hint">Check your JSON syntax and try again</div>' : ''}
            </div>
        `;
    }
});

// Global function to toggle features visibility
window.toggleFeatures = function () {
    const container = document.querySelector('.features-container');
    const toggle = document.querySelector('.features-toggle i');

    if (container.style.display === 'none') {
        container.style.display = 'block';
        toggle.classList.remove('fa-chevron-down');
        toggle.classList.add('fa-chevron-up');
    } else {
        container.style.display = 'none';
        toggle.classList.remove('fa-chevron-up');
        toggle.classList.add('fa-chevron-down');
    }
};