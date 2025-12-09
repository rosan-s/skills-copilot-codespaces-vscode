// ===== SCROLL FUNCTIONALITY =====
// Smooth scroll to section
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Update navigation active state on scroll
function updateActiveNavOnScroll() {
    const sections = document.querySelectorAll('.section');
    const navLinks = document.querySelectorAll('.nav-link');
    
    let currentSection = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (window.pageYOffset >= sectionTop - 150) {
            currentSection = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href').substring(1) === currentSection) {
            link.classList.add('active');
        }
    });
}

// Update scroll progress bar
function updateScrollProgress() {
    const windowHeight = window.innerHeight;
    const documentHeight = document.documentElement.scrollHeight;
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const scrollPercent = (scrollTop / (documentHeight - windowHeight)) * 100;
    
    document.getElementById('scroll-progress-bar').style.width = scrollPercent + '%';
}

// Reset form and scroll to detection section
function resetAndScroll() {
    resetForm();
    scrollToSection('detection');
    // Hide results
    document.getElementById('no-results-placeholder').style.display = 'block';
    document.getElementById('consensus-result').style.display = 'none';
    document.getElementById('model-predictions').style.display = 'none';
    document.getElementById('results-actions').style.display = 'none';
}

// Scroll event listener
window.addEventListener('scroll', () => {
    updateActiveNavOnScroll();
    updateScrollProgress();
});

// ===== MODEL COMPARISON =====
// Load and display model comparison
async function loadModelComparison() {
    try {
        console.log('Loading model comparison...');
        const modelsTable = document.getElementById('models-table');
        
        if (!modelsTable) {
            console.error('models-table element not found!');
            return;
        }
        
        const response = await fetch('/api/models');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Model data received:', data);

        if (data.success && data.models) {
            console.log('Displaying', data.models.length, 'models...');
            displayModelComparison(data.models, data.best_model);
            // Load performance graphs after table
            setTimeout(() => loadPerformanceGraphs(), 500);
        } else {
            console.error('API error - missing data:', data);
            modelsTable.innerHTML = 
                `<p class="error">⚠️ ${data.message || 'Error loading models'}</p>`;
        }
    } catch (error) {
        console.error('Fetch error:', error);
        const modelsTable = document.getElementById('models-table');
        if (modelsTable) {
            modelsTable.innerHTML = 
                `<p class="error">⚠️ Error loading models: ${error.message}</p>`;
        }
    }
}

// Load performance comparison graphs
async function loadPerformanceGraphs() {
    try {
        console.log('Fetching performance graphs...');
        const response = await fetch('/api/model-graphs');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Graph data received, has graph:', !!data.graph);
        
        if (data.success && data.graph) {
            console.log('Displaying performance graphs...');
            // Insert graph container after the table
            const modelsTable = document.getElementById('models-table');
            
            if (!modelsTable) {
                console.error('models-table element not found for appending graph!');
                return;
            }
            
            // Check if graph container already exists
            let graphContainer = document.getElementById('performance-graph-section');
            if (!graphContainer) {
                console.error('performance-graph-section element not found!');
                return;
            }
            
            graphContainer.innerHTML = `
                <h3 class="graph-title">
                    <span class="title-icon">📊</span>
                    Performance Metrics Visualization
                </h3>
                <div class="graph-image-container">
                    <img src="data:image/png;base64,${data.graph}" alt="Model Performance Graphs" class="performance-graph">
                </div>
            `;
            console.log('Performance graphs displayed successfully');
        } else {
            console.error('No graph data received:', data);
        }
    } catch (error) {
        console.error('Error loading performance graphs:', error);
    }
}

// Display model comparison table
function displayModelComparison(models, bestModel) {
    let tableHTML = `
        <div class="best-model-card">
            <div class="trophy-icon">🏆</div>
            <h3>Best Performing Model</h3>
            <div class="best-model-name">${bestModel}</div>
            <p>Recommended based on accuracy, precision, recall, and F1-score</p>
        </div>
        
        <div class="models-grid">
    `;

    models.forEach(model => {
        const isBest = model.name === bestModel;
        const rankClass = model.rank === 1 ? 'gold' : model.rank === 2 ? 'silver' : model.rank === 3 ? 'bronze' : '';
        
        tableHTML += `
            <div class="model-card ${isBest ? 'best-model' : ''}" onclick="showModelAnalysis('${model.name}', ${JSON.stringify(model).replace(/"/g, '&quot;')})">
                <div class="model-card-header">
                    <div class="rank-badge rank-${rankClass}">#${model.rank}</div>
                    ${isBest ? '<div class="star-badge">⭐</div>' : ''}
                </div>
                <h4 class="model-name">${model.name}</h4>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-label">Accuracy</div>
                        <div class="metric-value ${getMetricClass(model.accuracy)}">${model.accuracy}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: ${model.accuracy}%"></div>
                        </div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Precision</div>
                        <div class="metric-value ${getMetricClass(model.precision)}">${model.precision}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: ${model.precision}%"></div>
                        </div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Recall</div>
                        <div class="metric-value ${getMetricClass(model.recall)}">${model.recall}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: ${model.recall}%"></div>
                        </div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">F1-Score</div>
                        <div class="metric-value ${getMetricClass(model.f1_score)}">${model.f1_score}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: ${model.f1_score}%"></div>
                        </div>
                    </div>
                </div>
                <div class="view-analysis-hint">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <polyline points="12 16 16 12 12 8"></polyline>
                        <line x1="8" y1="12" x2="16" y2="12"></line>
                    </svg>
                    Click for detailed analysis
                </div>
            </div>
        `;
    });

    tableHTML += `
        </div>
        
        <div id="performance-graph-section" class="graph-section">
            <div class="loading-graph">Loading performance visualizations...</div>
        </div>
    `;

    document.getElementById('models-table').innerHTML = tableHTML;
}

// Get CSS class based on metric value
function getMetricClass(value) {
    if (value >= 90) return 'metric-excellent';
    if (value >= 80) return 'metric-good';
    if (value >= 70) return 'metric-fair';
    return 'metric-poor';
}

function showModelAnalysis(modelName, modelData) {
    const modal = document.getElementById('model-analysis-modal');
    const content = document.getElementById('modal-analysis-content');
    
    // Fetch detailed analysis from backend
    fetch('/api/model-analysis', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_name: modelName })
    })
    .then(response => response.json())
    .then(data => {
        content.innerHTML = generateAnalysisHTML(modelName, modelData, data);
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    })
    .catch(error => {
        console.error('Error fetching model analysis:', error);
        content.innerHTML = generateBasicAnalysisHTML(modelName, modelData);
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    });
}

function closeAnalysisModal() {
    const modal = document.getElementById('model-analysis-modal');
    modal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

function generateAnalysisHTML(modelName, modelData, detailedData) {
    return `
        <div class="analysis-header">
            <h2>${modelName} Analysis</h2>
            <div class="analysis-rank">Rank #${modelData.rank}</div>
        </div>
        
        <div class="analysis-sections">
            <div class="analysis-section">
                <h3>📊 Performance Metrics</h3>
                <div class="metrics-detailed">
                    <div class="metric-detail">
                        <div class="metric-detail-label">
                            <strong>Accuracy</strong>
                            <span class="metric-tooltip">Overall correctness of predictions</span>
                        </div>
                        <div class="metric-detail-bar">
                            <div class="metric-detail-fill" style="width: ${modelData.accuracy}%; background: linear-gradient(90deg, #4CAF50, #8BC34A);"></div>
                            <span class="metric-detail-value">${modelData.accuracy}%</span>
                        </div>
                    </div>
                    <div class="metric-detail">
                        <div class="metric-detail-label">
                            <strong>Precision</strong>
                            <span class="metric-tooltip">Accuracy of positive predictions</span>
                        </div>
                        <div class="metric-detail-bar">
                            <div class="metric-detail-fill" style="width: ${modelData.precision}%; background: linear-gradient(90deg, #2196F3, #03A9F4);"></div>
                            <span class="metric-detail-value">${modelData.precision}%</span>
                        </div>
                    </div>
                    <div class="metric-detail">
                        <div class="metric-detail-label">
                            <strong>Recall</strong>
                            <span class="metric-tooltip">Ability to find all positive cases</span>
                        </div>
                        <div class="metric-detail-bar">
                            <div class="metric-detail-fill" style="width: ${modelData.recall}%; background: linear-gradient(90deg, #FF9800, #FFC107);"></div>
                            <span class="metric-detail-value">${modelData.recall}%</span>
                        </div>
                    </div>
                    <div class="metric-detail">
                        <div class="metric-detail-label">
                            <strong>F1-Score</strong>
                            <span class="metric-tooltip">Harmonic mean of precision and recall</span>
                        </div>
                        <div class="metric-detail-bar">
                            <div class="metric-detail-fill" style="width: ${modelData.f1_score}%; background: linear-gradient(90deg, #9C27B0, #E91E63);"></div>
                            <span class="metric-detail-value">${modelData.f1_score}%</span>
                        </div>
                    </div>
                </div>
            </div>
            
            ${detailedData.confusion_matrix ? `
            <div class="analysis-section">
                <h3>🎯 Confusion Matrix</h3>
                <div class="confusion-matrix">
                    <table>
                        <tr>
                            <th></th>
                            <th>Predicted: No PCOS</th>
                            <th>Predicted: PCOS</th>
                        </tr>
                        <tr>
                            <th>Actual: No PCOS</th>
                            <td class="tn">${detailedData.confusion_matrix.tn || 'N/A'}</td>
                            <td class="fp">${detailedData.confusion_matrix.fp || 'N/A'}</td>
                        </tr>
                        <tr>
                            <th>Actual: PCOS</th>
                            <td class="fn">${detailedData.confusion_matrix.fn || 'N/A'}</td>
                            <td class="tp">${detailedData.confusion_matrix.tp || 'N/A'}</td>
                        </tr>
                    </table>
                </div>
            </div>
            ` : ''}
            
            <div class="analysis-section">
                <h3>💡 Model Insights</h3>
                <div class="insights-grid">
                    <div class="insight-card">
                        <div class="insight-icon">✅</div>
                        <h4>Strengths</h4>
                        <ul>
                            ${generateStrengths(modelData)}
                        </ul>
                    </div>
                    <div class="insight-card">
                        <div class="insight-icon">⚠️</div>
                        <h4>Considerations</h4>
                        <ul>
                            ${generateConsiderations(modelData)}
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="analysis-section">
                <h3>🔍 Model Description</h3>
                <div class="model-description">
                    ${getModelDescription(modelName)}
                </div>
            </div>
        </div>
    `;
}

function generateBasicAnalysisHTML(modelName, modelData) {
    return generateAnalysisHTML(modelName, modelData, {});
}

function generateStrengths(modelData) {
    const strengths = [];
    if (modelData.accuracy >= 90) strengths.push('<li>Excellent overall accuracy</li>');
    if (modelData.precision >= 90) strengths.push('<li>High precision in positive predictions</li>');
    if (modelData.recall >= 90) strengths.push('<li>Strong ability to detect PCOS cases</li>');
    if (modelData.f1_score >= 90) strengths.push('<li>Well-balanced performance</li>');
    
    if (strengths.length === 0) {
        if (modelData.accuracy >= 80) strengths.push('<li>Good overall performance</li>');
        if (modelData.precision >= 80) strengths.push('<li>Reliable positive predictions</li>');
    }
    
    return strengths.length > 0 ? strengths.join('') : '<li>Moderate performance across metrics</li>';
}

function generateConsiderations(modelData) {
    const considerations = [];
    if (modelData.accuracy < 80) considerations.push('<li>May benefit from additional training</li>');
    if (modelData.precision < 85) considerations.push('<li>Some false positives may occur</li>');
    if (modelData.recall < 85) considerations.push('<li>May miss some PCOS cases</li>');
    if (Math.abs(modelData.precision - modelData.recall) > 10) {
        considerations.push('<li>Imbalanced precision-recall trade-off</li>');
    }
    
    return considerations.length > 0 ? considerations.join('') : '<li>Well-balanced performance metrics</li>';
}

function getModelDescription(modelName) {
    const descriptions = {
        'Logistic Regression': 'A linear model that predicts PCOS likelihood using weighted combinations of input features. Fast and interpretable, making it ideal for understanding which factors contribute most to diagnosis.',
        'Random Forest': 'An ensemble method combining multiple decision trees to make robust predictions. Excellent at handling complex interactions between symptoms and hormonal factors.',
        'Support Vector Machine': 'A powerful algorithm that finds optimal boundaries between PCOS and non-PCOS cases in high-dimensional space. Particularly effective with complex, non-linear patterns.',
        'XGBoost': 'An advanced gradient boosting algorithm that builds trees sequentially to correct previous errors. Achieves the highest accuracy for PCOS detection with strong predictive power.',
        'Gradient Boosting': 'An advanced gradient boosting algorithm that builds trees sequentially to correct previous errors. Achieves the highest accuracy for PCOS detection with strong predictive power.'
    };
    
    return descriptions[modelName] || 'A machine learning model trained to detect PCOS based on clinical, hormonal, and symptomatic indicators.';
}

// Form submission handler - handles prediction request
async function handlePredictionSubmit(e) {
    console.log('Form submitted!');
    e.preventDefault();

    // Collect form data
    const formData = {};
    
    // Numeric inputs
    const numericFields = ['Age', 'BMI', 'Cycle_length', 'FSH', 'LH', 'TSH', 'AMH', 'Insulin'];
    numericFields.forEach(field => {
        formData[field] = parseFloat(document.getElementById(field).value);
    });

    // Checkbox inputs (binary)
    const checkboxFields = ['Weight_gain', 'Hair_growth', 'Skin_darkening', 'Hair_loss', 
                           'Pimples', 'Fast_food', 'Reg_Exercise'];
    checkboxFields.forEach(field => {
        formData[field] = document.getElementById(field).checked ? 1 : 0;
    });
    
    console.log('Form data collected:', formData);

    // Scroll to results section
    scrollToSection('results');
    
    // Show loading state
    document.getElementById('no-results-placeholder').style.display = 'none';
    document.getElementById('consensus-result').style.display = 'block';
    document.getElementById('consensus-result').innerHTML = '<div class="spinner"></div><p>Analyzing with 4 ML models... (5-10 seconds)</p>';
    document.getElementById('model-predictions').style.display = 'none';

    try {
        // Create abort controller with 15 second timeout (fast sklearn models only)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            controller.abort();
        }, 15000);

        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        // Check if response is ok
        if (!response.ok) {
            if (response.status === 502) {
                throw new Error(`Server timeout (502). Analysis is taking longer than expected. Please try again in a moment.`);
            } else if (response.status === 503) {
                throw new Error(`Server busy (503). Models may still be loading. Please wait and try again.`);
            } else if (response.status === 504) {
                throw new Error(`Gateway timeout (504). Please try again.`);
            } else {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
        }

        // Check if response has content
        const contentType = response.headers.get("content-type");
        if (!contentType || !contentType.includes("application/json")) {
            throw new Error("Response is not JSON");
        }

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            document.getElementById('consensus-result').innerHTML = 
                `<p class="error">⚠️ Error: ${data.message || 'Unknown error occurred'}</p>`;
        }
    } catch (error) {
        console.error('Prediction error:', error);
        
        let errorMessage = error.message;
        if (error.name === 'AbortError') {
            errorMessage = 'Request timeout. The server is taking too long to respond. This sometimes happens on first request or during peak usage. Please try again.';
        }
        
        document.getElementById('consensus-result').innerHTML = 
            `<p class="error">⚠️ Error making prediction: ${errorMessage}<br><br>Please check that all fields are filled correctly and try again. If the problem persists, wait a moment and try again.</p>`;
    }
}

// Display prediction results
function displayResults(data) {
    // Display consensus result
    const consensus = data.consensus;
    const consensusClass = consensus.prediction === 1 ? 'consensus-positive' : 'consensus-negative';
    
    document.getElementById('consensus-result').innerHTML = `
        <h3>🎯 Consensus Prediction</h3>
        <div class="consensus-result ${consensusClass}">
            ${consensus.result}
        </div>
        <div class="confidence-score">${consensus.votes}</div>
    `;
    document.getElementById('consensus-result').style.display = 'block';

    // Display individual model predictions
    const predictionsHTML = Object.entries(data.predictions)
        .map(([modelName, prediction]) => {
            if (prediction.error) {
                return `
                    <div class="model-card">
                        <h4>${modelName}</h4>
                        <p class="error">Error: ${prediction.error}</p>
                    </div>
                `;
            }

            const resultClass = prediction.prediction === 1 ? 'prediction-positive' : 'prediction-negative';
            const icon = prediction.prediction === 1 ? '⚠️' : '✅';

            return `
                <div class="model-card">
                    <h4>${icon} ${modelName}</h4>
                    <div class="prediction-badge ${resultClass}">
                        ${prediction.result}
                    </div>
                    <div class="accuracy-bar">
                        <div class="accuracy-label">
                            <span>Confidence</span>
                            <span>${prediction.confidence.toFixed(1)}%</span>
                        </div>
                        <div class="bar-container">
                            <div class="bar-fill" style="width: ${prediction.confidence}%"></div>
                        </div>
                    </div>
                    ${prediction.accuracy ? `
                    <div class="accuracy-bar" style="margin-top: 10px;">
                        <div class="accuracy-label">
                            <span>Model Accuracy</span>
                            <span>${prediction.accuracy}%</span>
                        </div>
                        <div class="bar-container">
                            <div class="bar-fill" style="width: ${prediction.accuracy}%"></div>
                        </div>
                    </div>
                    ` : ''}
                </div>
            `;
        })
        .join('');

    document.getElementById('model-predictions').innerHTML = predictionsHTML;
    document.getElementById('model-predictions').style.display = 'grid';
    document.getElementById('results-actions').style.display = 'flex';
}

// Reset form
function resetForm() {
    document.getElementById('prediction-form').reset();
    
    // Set default values
    document.getElementById('Age').value = 25;
    document.getElementById('BMI').value = 24.5;
    document.getElementById('Cycle_length').value = 28;
    document.getElementById('FSH').value = 5.5;
    document.getElementById('LH').value = 8.5;
    document.getElementById('TSH').value = 2.2;
    document.getElementById('AMH').value = 4.5;
    document.getElementById('Insulin').value = 12.5;
}

// Show detailed model analysis in modal
async function showModelDetails(modelName, modelData) {
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.onclick = function(e) {
        if (e.target === modal) {
            document.body.removeChild(modal);
        }
    };
    
    const strengths = modelData.strengths || [];
    const weaknesses = modelData.weaknesses || [];
    
    // Fetch confusion matrix graph
    let confusionMatrixHTML = '<p>Loading confusion matrix...</p>';
    
    try {
        const response = await fetch(`/api/confusion-matrix/${encodeURIComponent(modelName)}`);
        const data = await response.json();
        
        if (data.success && data.graph) {
            confusionMatrixHTML = `
                <img src="data:image/png;base64,${data.graph}" alt="Confusion Matrix" class="confusion-matrix-graph">
            `;
        } else {
            confusionMatrixHTML = '<p>Confusion matrix not available</p>';
        }
    } catch (error) {
        confusionMatrixHTML = '<p>Error loading confusion matrix</p>';
    }
    
    modal.innerHTML = `
        <div class="modal-content">
            <span class="modal-close" onclick="this.parentElement.parentElement.remove()">&times;</span>
            <h2>📊 ${modelName} - Detailed Analysis</h2>
            
            <div class="modal-section">
                <h3>Performance Metrics</h3>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">${modelData.accuracy}%</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${modelData.precision}%</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${modelData.recall}%</div>
                        <div class="metric-label">Recall</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${modelData.f1_score}%</div>
                        <div class="metric-label">F1-Score</div>
                    </div>
                </div>
            </div>
            
            <div class="modal-section">
                <h3>📈 Confusion Matrix</h3>
                <div class="confusion-matrix-container">
                    ${confusionMatrixHTML}
                </div>
            </div>
            
            <div class="modal-section">
                <h3>✓ Strengths</h3>
                <ul class="analysis-list strengths-list">
                    ${strengths.map(s => `<li>${s}</li>`).join('')}
                </ul>
            </div>
            
            
            <div class="modal-section">
                <h3>⚠ Limitations</h3>
                <ul class="analysis-list weaknesses-list">
                    ${weaknesses.map(w => `<li>${w}</li>`).join('')}
                </ul>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
}

// Download research paper
function downloadResearchPaper() {
    // Show loading indicator
    const button = event.target.closest('.cta-button');
    const originalContent = button.innerHTML;
    button.innerHTML = '⏳ Downloading...';
    button.disabled = true;
    
    // Trigger download
    window.location.href = '/api/download-paper';
    
    // Reset button after delay
    setTimeout(() => {
        button.innerHTML = originalContent;
        button.disabled = false;
    }, 2000);
}

// Dataset upload functionality
let selectedFile = null;

// Load model comparison and setup event listeners on page load
document.addEventListener('DOMContentLoaded', function() {
    // Load model comparison
    loadModelComparison();
    
    // Attach form submission handler
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePredictionSubmit);
    }
    
    // Initialize scroll progress
    updateScrollProgress();
    updateActiveNavOnScroll();
});