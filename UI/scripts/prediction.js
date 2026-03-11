/**
 * Main prediction interface controller for Dr. PROBEN web application.
 * 
 * Manages dual-mode prediction interface (single vs batch) with disease-specific
 * configurations, real-time form validation, chart visualization, and CSV batch processing.
 * 
 * Key Features:
 * - Dynamic form generation based on disease type (diabetes/heart/cancer)
 * - Interactive Chart.js doughnut visualization for single predictions
 * - CSV upload validation with automatic data cleaning
 * - Batch prediction with downloadable results
 * - Session state management across page navigation
 * - Theme-aware UI with disease-specific color schemes
 * 
 * Architecture:
 * - Single Prediction: Form → Validation → API → Chart Update
 * - Batch Prediction: CSV Upload → Validation → API → Table Display → Download
 * 
 * Dependencies:
 * - Chart.js for data visualization
 * - SweetAlert2 for modals and notifications
 * - Tippy.js for tooltips
 */

import { diseaseStates, diseaseConfigs, diseaseInfoContent } from "./diseaseConfig.js";

document.addEventListener('DOMContentLoaded', () => {
    // Global state management
    let currentDisease = 'diabetes';
    let currentMode = 'single';
    
    // =====================================================================
    // SHARED UTILITIES & HELPERS
    // =====================================================================

    // Inline accepted-range helper (shown near the focused input)
    function toFiniteNumber(value) {
        if (value === undefined || value === null) return null;
        if (typeof value === 'number') return Number.isFinite(value) ? value : null;
        if (typeof value !== 'string') return null;
    
        const trimmed = value.trim();
        if (!trimmed) return null;
        const n = Number(trimmed);
        return Number.isFinite(n) ? n : null;
        }
    
        function formatAcceptedRange(attr) {
        const min = toFiniteNumber(attr.min);
        const max = toFiniteNumber(attr.max);
    
        const parts = [];
        if (min !== null) parts.push(`Min: ${min}`);
        if (max !== null) parts.push(`Max: ${max}`);
    
        // Only show explicit min/max values (no placeholder inference)
        if (parts.length > 0) return parts.join(' | ');
        return 'Min/Max not specified.';
        
        }
    
        function ensureInlineRangeHelper() {
            let helper = document.getElementById('inlineRangeHelper');
            if (helper) return helper;
    
            helper = document.createElement('div');
            helper.id = 'inlineRangeHelper';
            helper.style.display = 'none';
            helper.style.position = 'fixed';
            helper.style.zIndex = '9999';
            helper.style.maxWidth = '260px';
            helper.style.padding = '10px 12px';
            helper.style.borderRadius = '10px';
            helper.style.background = '#ffffff';
            helper.style.border = '1px solid rgba(0,0,0,0.12)';
            helper.style.boxShadow = '0 6px 18px rgba(0,0,0,0.12)';
            helper.style.fontFamily = "'Poppins', sans-serif";
            helper.style.fontSize = '13px';
            helper.style.color = '#333';
    
            document.body.appendChild(helper);
            return helper;
        }
    
        function showInlineRangeHelperForInput(input, attr) {
            const helper = ensureInlineRangeHelper();
            const rangeText = formatAcceptedRange(attr);
    
            helper.innerHTML = `
                <div style="font-weight:600; margin-bottom:4px;">Accepted values</div>
                <div>${rangeText}</div>
            `;
    
            // Position to the right of the input, fallback above if near the edge
            const rect = input.getBoundingClientRect();
            const margin = 10;
            const desiredLeft = rect.right + margin;
            const desiredTop = rect.top;
    
            helper.style.display = 'block';
            const helperRect = helper.getBoundingClientRect();
            const fitsRight = (desiredLeft + helperRect.width) <= (window.innerWidth - margin);
    
            const left = fitsRight ? desiredLeft : Math.max(margin, rect.left);
            const top = fitsRight
                ? Math.min(window.innerHeight - helperRect.height - margin, desiredTop)
                : Math.max(margin, rect.top - helperRect.height - margin);
    
            helper.style.left = `${left}px`;
            helper.style.top = `${top}px`;
            helper.style.outline = `2px solid ${getThemeColor()}22`;
        }
    
        function hideInlineRangeHelper() {
            const helper = document.getElementById('inlineRangeHelper');
            if (helper) helper.style.display = 'none';
        }
    
        function attachInlineRangeHelper(config) {
            // Only for single prediction, and only for diabetes + heart
            if (currentMode !== 'single' || !config || !['diabetes', 'heart'].includes(currentDisease)) {
                hideInlineRangeHelper();
                return;
            }
    
            // Shared timer so blur from one input doesn't hide the helper after another input is focused
            if (window.__inlineRangeHideTimerId) {
                clearTimeout(window.__inlineRangeHideTimerId);
                window.__inlineRangeHideTimerId = null;
            }
    
            config.attributes.forEach(attr => {
                if (attr.type !== 'number') return;
                const input = document.getElementById(attr.id);
                if (!input) return;
    
                // Avoid attaching duplicate listeners when re-rendering
                if (input.dataset.inlineRangeAttached === 'true') return;
                input.dataset.inlineRangeAttached = 'true';
    
                input.addEventListener('focus', () => {
                    if (window.__inlineRangeHideTimerId) {
                        clearTimeout(window.__inlineRangeHideTimerId);
                        window.__inlineRangeHideTimerId = null;
                    }
                    showInlineRangeHelperForInput(input, attr);
                });
    
                input.addEventListener('input', () => showInlineRangeHelperForInput(input, attr));
    
                input.addEventListener('blur', () => {
                    // Delay hide so tabbing/clicking to another input doesn't flicker
                    window.__inlineRangeHideTimerId = setTimeout(() => {
                        // Only hide if focus didn't move to another input
                        const active = document.activeElement;
                        if (!active || active.tagName !== 'INPUT') {
                            hideInlineRangeHelper();
                        }
                        window.__inlineRangeHideTimerId = null;
                    }, 50);
                });
            });
    
            // Global listeners (attach once)
            if (!document.body.dataset.inlineRangeGlobalAttached) {
                document.body.dataset.inlineRangeGlobalAttached = 'true';
    
                document.addEventListener('click', (e) => {
                    const helper = document.getElementById('inlineRangeHelper');
                    if (!helper) return;
                    if (helper.contains(e.target)) return;
                    if (e.target && e.target.tagName === 'INPUT') return;
                    hideInlineRangeHelper();
                });
    
                window.addEventListener('resize', () => hideInlineRangeHelper());
                window.addEventListener('scroll', () => hideInlineRangeHelper(), true);
            }
        }

    function initializePage() {
        // Check for stored disease and mode from sessionStorage
        const storedDisease = sessionStorage.getItem('selectedDisease');
        const storedMode = sessionStorage.getItem('selectedMode');
        
        if (storedDisease && diseaseConfigs[storedDisease]) {
            currentDisease = storedDisease;
        }
        
        if (storedMode && (storedMode === 'single' || storedMode === 'batch')) {
            currentMode = storedMode;
        }

        // Initialize single and batch prediction FIRST to expose functions
        initializeSinglePrediction();
        initializeBatchPrediction();
        
        // THEN update page content (which calls chart color updates)
        updatePageContent(currentDisease);
        
        // Setup event listeners
        setupEventListeners();
        
        // Show the correct mode section
        switchMode(currentMode);
        
        // Add beforeunload event listener to warn about unsaved data
        window.addEventListener('beforeunload', handleBeforeUnload);
    }

    function setupEventListeners() {

        document.querySelectorAll('.segment-button').forEach(button => {
            button.addEventListener('click', function() {
                const mode = this.dataset.mode;
                if (mode !== currentMode) {
                    switchMode(mode);
                }
            });
        });
    }

    function switchMode(mode) {
        currentMode = mode;
    
        // Store current mode in sessionStorage
        sessionStorage.setItem('selectedMode', mode);
        
        // Update segment control UI
        document.querySelectorAll('.segment-button').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
        
        // Show/hide sections
        document.getElementById('singlePredictionSection').classList.toggle('active', mode === 'single');
        document.getElementById('batchPredictionSection').classList.toggle('active', mode === 'batch');
    }

    function updatePageContent(disease) {
        currentDisease = disease;
        const config = diseaseConfigs[disease];
        
        // Store current disease in sessionStorage
        sessionStorage.setItem('selectedDisease', disease);
        
        // Update title
        document.getElementById('diseaseTitle').textContent = config.name + " Classifier";
        
        // Apply theme
        document.body.className = `theme-${disease}`;
        
        // Update single prediction form
        updateSinglePredictionForm(config);
        
        // Update batch prediction table
        updateBatchPredictionTable(config);

        // Update chart colors and legend after singlePrediction is initialized
        if (window.singlePrediction) {
            window.singlePrediction.updateChartColors(disease);
        }
    }

    // Back button functionality
    window.goBack = function() {
        // Clear sessionStorage when intentionally leaving the page
        sessionStorage.removeItem('selectedDisease');
        sessionStorage.removeItem('selectedMode');
        window.location.href = '../index.html';
    };

    function handleBeforeUnload(e) {
        e.preventDefault();
        e.returnValue = ''; // Chrome requires returnValue to be set
        return ''; // Some browsers show this message
        
    }

    
    // =====================================================================
    // SINGLE PREDICTION MODE
    // =====================================================================

    function initializeSinglePrediction() {
        const form = document.getElementById('predictionForm');
        const displayResult = document.getElementById('result');
        let percentText = document.getElementById('percentText');
        let chartLabel = document.getElementById('chartLabel');
        let percentage = 0;

        // Initialize chart data
        const data = {
            datasets: [{
                data: [percentage, 100 - percentage],
                backgroundColor: ['#096F29', '#929292'],
                borderWidth: 0,
                cutout: '65%',
            }]
        };

        const shadowPlugin = {
            id: 'shadow',
            beforeDatasetDraw(chart, args) {
                const ctx = chart.ctx;
                ctx.save();
                ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
                ctx.shadowBlur = 10;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 4;
            },
            afterDatasetDraw(chart) {
                chart.ctx.restore();
            }
        };

        const config = {
            type: 'doughnut',
            data: data,
            options: {
                responsive: true,
                animation: {
                    duration: 0,
                    animateRotate: true,
                    animateScale: false,
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        enabled: true,
                        position: 'nearest',
                        z: 9999, // Highest z-index
                        callbacks: {
                            label: function(context) {
                                const dataIndex = context.dataIndex;
                                const value = context.parsed;
                                const config = diseaseConfigs[currentDisease];

                                if (dataIndex === 0) {
                                    return `${config.positiveClass}: ${Math.round(value)}%`;
                                } else {
                                    return `${config.negativeClass}: ${Math.round(value)}%`;
                                }
                            }
                        }
                    },
                },
                layout: {
                    padding: 5
                }
            },
            plugins: [shadowPlugin]
        };

        const chart = new Chart(document.getElementById('doughnutChart'), config);

        function updateChartColors(disease) {
            const themeColors = {
                diabetes: ['#096F29', '#929292'],
                heart: ['#811111', '#929292'],
                cancer: ['#1E2F4E', '#929292']
            };

            chart.data.datasets[0].backgroundColor = themeColors[disease];
            chart.update('none');
            
            updateCustomLegend(disease);
            updateLegendColors(disease);
        }

        function updateCustomLegend(disease) {
            const config = diseaseConfigs[disease];
            const state = diseaseStates[disease];
            const legendContainer = document.getElementById('chartLegend');
            
            if (!legendContainer) return;
            
            // Update legend labels with current disease class names
            const legendPositive = legendContainer.querySelector('.legend-positive .legend-label');
            const legendNegative = legendContainer.querySelector('.legend-negative .legend-label');
            
            if (legendPositive && legendNegative) {
                // Check if we have result data with percentage
                if (state.singleResultData && typeof state.singleResultData.percentage !== 'undefined') {
                    const percentage = state.singleResultData.percentage;
                    const prediction = state.singleResultData.prediction;
                    
                    // Calculate percentages based on prediction
                    let positivePercent, negativePercent;
                    if (prediction === 1) {
                        // Positive class was predicted
                        positivePercent = Math.round(percentage);
                        negativePercent = Math.round(100 - percentage);
                    } else {
                        // Negative class was predicted
                        negativePercent = Math.round(percentage);
                        positivePercent = Math.round(100 - percentage);
                    }
                    
                    legendPositive.textContent = `${config.positiveClass}: ${positivePercent}%`;
                    legendNegative.textContent = `${config.negativeClass}: ${negativePercent}%`;
                } else {
                    // No result data, show class names only
                    legendPositive.textContent = config.positiveClass;
                    legendNegative.textContent = config.negativeClass;
                }
            }
        }

        // Function to update legend colors based on theme
        function updateLegendColors(disease) {
            const themeColors = {
                diabetes: ['#096F29', '#929292'],
                heart: ['#811111', '#929292'],
                cancer: ['#1E2F4E', '#929292']
            };
            
            const colors = themeColors[disease];
            const legendPositive = document.querySelector('.legend-positive .legend-color');
            const legendNegative = document.querySelector('.legend-negative .legend-color');
            
            if (legendPositive && legendNegative) {
                legendPositive.style.backgroundColor = colors[0];
                legendNegative.style.backgroundColor = colors[1];
            }
        }

        // Validation helper functions
        function addErrorBorder(element, attr) {
            console.log('Adding error border to:', attr.id);
            if (attr.type === 'radio') {
                const radioGroup = element.closest('.radio-group');
                if (radioGroup) {
                    radioGroup.classList.add('input-error');
                    console.log('Added error to radio group');
                }
            } else {
                element.classList.add('input-error');
                console.log('Added error to element');
            }
        }

        function removeErrorBorder(element, attr) {
            console.log('Removing error border from:', attr.id);
            if (attr.type === 'radio') {
                const radioGroup = element.closest('.radio-group');
                if (radioGroup) {
                    radioGroup.classList.remove('input-error');
                }
            } else {
                element.classList.remove('input-error');
            }
        }

        function validateForm() {
            const config = diseaseConfigs[currentDisease];
            let isValid = true;
            const invalidFields = [];

            console.log('Validating form for disease:', currentDisease);

            config.attributes.forEach(attr => {
                let element;
                let hasValue = false;

                if (attr.type === 'radio') {
                    element = document.querySelector(`input[name="${attr.id}"]`);
                    const checked = document.querySelector(`input[name="${attr.id}"]:checked`);
                    hasValue = !!checked;
                } else if (attr.type === 'dropdown') {
                    element = document.getElementById(attr.id);
                    hasValue = element && element.value !== '' && element.selectedIndex !== 0;
                } else {
                    element = document.getElementById(attr.id);
                    hasValue = element && element.value.trim() !== '';
                }

                console.log(`Field ${attr.id}: hasValue=${hasValue}, element=`, element);

                if (!hasValue) {
                    addErrorBorder(element, attr);
                    invalidFields.push(attr.label);
                    isValid = false;
                } else {
                    removeErrorBorder(element, attr);
                }
            });

            console.log('Validation result:', { isValid, invalidFields });
            return { isValid, invalidFields };
        }

        function setupRealTimeValidation() {
            const config = diseaseConfigs[currentDisease];
            console.log('setupRealTimeValidation called for:', currentDisease);
            console.log('Number of attributes:', config.attributes.length);

            config.attributes.forEach(attr => {

                console.log('Setting up listener for:', attr.id, 'type:', attr.type);
                
                if (attr.type === 'radio') {
                    const radioInputs = document.querySelectorAll(`input[name="${attr.id}"]`);
                    console.log('Found radio inputs:', radioInputs.length);
                    radioInputs.forEach(radio => {
                        radio.addEventListener('change', () => {
                            console.log('Radio changed:', attr.id);
                            removeErrorBorder(radio, attr);
                        });
                    });
                } else if (attr.type === 'dropdown') {
                    const element = document.getElementById(attr.id);
                    if (element) {
                        console.log('Found dropdown:', attr.id);
                        element.addEventListener('change', () => {
                            console.log('Dropdown changed:', attr.id);
                            removeErrorBorder(element, attr);
                        });
                    } else {
                        console.error('Dropdown not found:', attr.id);
                    }
                } else {
                    const element = document.getElementById(attr.id);
                    if (element) {
                        console.log('Found input:', attr.id);
                        element.addEventListener('input', () => {
                            console.log('Input changed:', attr.id, 'value:', element.value);
                            removeErrorBorder(element, attr);
                        });
                    } else {
                        console.error('Input not found:', attr.id);
                    }
                }
            });
        }

        // Handle form submission
        form.addEventListener('submit', async function(e) {
            e.preventDefault();

            const config = diseaseConfigs[currentDisease];

            // Validate form
            console.log('Form submitted, validating...');
            const validation = validateForm();

            if (!validation.isValid) {
                console.log('Validation failed, invalid fields:', validation.invalidFields);
                
                Swal.fire({
                    icon: 'error',
                    title: 'Incomplete Form',
                    html: `
                        <div style="text-align: center;">
                            <p style="font-size: 18px; line-height: 1.8;">
                                Please complete all required fields before submitting the form.
                            </p>
                            <p style="margin-top: 15px; font-size: 14px; color: #666;">
                                Fields with <span style="color: #ff0000; font-weight: bold;">red borders</span> need to be filled or corrected.
                            </p>
                        </div>
                    `,
                    confirmButtonText: 'OK',
                    confirmButtonColor: getThemeColor(),
                    width: '450px'
                });
                return;
            }

            console.log('Validation passed, proceeding with prediction...');

            // Get form data dynamically
            const formData = {};
            config.attributes.forEach(attr => {
                if (attr.type === 'radio') {
                    // For radio buttons, get the checked value
                    const radioInput = document.querySelector(`input[name="${attr.id}"]:checked`);
                    if (radioInput) {
                        formData[attr.id] = radioInput.value;
                    }
                } else if (attr.type === 'dropdown') {
                    // For dropdowns, get the selected value
                    const selectInput = document.getElementById(attr.id);
                    if (selectInput) {
                        formData[attr.id] = selectInput.value;
                    }
                } else {
                    // For regular inputs (text, number)
                    const input = document.getElementById(attr.id);
                    if (input) {
                        formData[attr.id] = input.value;
                    }
                }
            });

            console.log('Form data:', formData);

            try {
                const response = await fetch(`http://127.0.0.1:5000${config.endpoint}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });

                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                const result = await response.json();

                console.log('API Response:', result); // Debug log

                // Store result
                diseaseStates[currentDisease].singleResultData = {
                    prediction: result.prediction,
                    percentage: result.percentage
                };

                // Hide placeholder and show results
                document.getElementById('resultPlaceholder').style.display = 'none';
                document.getElementById('resultContent').style.display = 'flex';
                
                const resultElement = document.querySelector('#resultContent h1');
                const descElement = document.querySelector('#resultContent p');

                // Display the predicted class name and description
                if (result.prediction === 1) {
                    resultElement.textContent = config.positiveClass;
                    descElement.textContent = config.positiveDesc;
                } else {
                    resultElement.textContent = config.negativeClass;
                    descElement.textContent = config.negativeDesc;
                }

                // Update center percentage - use result.percentage which is already 0-100
                percentText.textContent = `${Math.round(result.percentage)}%`;
                
                // Calculate chart percentages
                // result.percentage is the confidence (0-100) of whichever class was predicted
                let positivePercent, negativePercent;
                
                if (result.prediction === 1) {
                    // Positive class was predicted
                    // So result.percentage = confidence of positive class
                    positivePercent = result.percentage;
                    negativePercent = 100 - result.percentage;
                } else {
                    // Negative class was predicted (prediction === 0)
                    // So result.percentage = confidence of negative class
                    negativePercent = result.percentage;
                    positivePercent = 100 - result.percentage;
                }
                
                console.log('Chart data - Positive:', positivePercent, 'Negative:', negativePercent); // Debug log
                
                // Update chart - array is ALWAYS [positive%, negative%]
                chart.data.datasets[0].data = [positivePercent, negativePercent];
                chart.update('active');
                
                // Update legend with both percentages
                updateCustomLegend(currentDisease);

            } catch (error) {
                console.error('Prediction error:', error);
                Swal.fire({
                    icon: 'error',
                    title: 'Prediction Failed',
                    text: 'Unable to process the prediction. Please try again.',
                    confirmButtonColor: getThemeColor()
                });
            }
        });

        // Handle form reset
        form.addEventListener('reset', () => {
            diseaseStates[currentDisease].singleFormData = {};
            diseaseStates[currentDisease].singleResultData = null;

            setTimeout(() => {
                const config = diseaseConfigs[currentDisease];
                
                // Reset all form fields and remove error borders
                config.attributes.forEach(attr => {
                    if (attr.type === 'radio') {
                        const radioInputs = document.querySelectorAll(`input[name="${attr.id}"]`);
                        radioInputs.forEach(radio => {
                            radio.checked = false;
                            removeErrorBorder(radio, attr);
                        });
                    } else if (attr.type === 'dropdown') {
                        const selectInput = document.getElementById(attr.id);
                        if (selectInput) {
                            selectInput.selectedIndex = 0;
                            removeErrorBorder(selectInput, attr);
                        }
                    } else {
                        const input = document.getElementById(attr.id);
                        if (input) {
                            input.value = '';
                            removeErrorBorder(input, attr);
                        }
                    }
                });
            }, 0);

            // Show placeholder and hide results
            document.getElementById('resultPlaceholder').style.display = 'flex';
            document.getElementById('resultContent').style.display = 'none';
            
            percentText.textContent = '--%';
            chart.data.datasets[0].data = [0, 100];
            chart.update('none');
            
            // Reset legend to show class names only
            updateCustomLegend(currentDisease);
        });

        // Expose functions for global access
        window.singlePrediction = {
            updateChartColors,
            updateCustomLegend,
            updateLegendColors,
            setupRealTimeValidation,
            chart
        };

        console.log('singlePrediction object created:', window.singlePrediction);
    }

    function updateSinglePredictionForm(config) {
        const labelsContainer = document.querySelector('.labels-container');
        const inputsContainer = document.querySelector('.inputs-container');

        labelsContainer.innerHTML = '';
        inputsContainer.innerHTML = '';

        config.attributes.forEach(attr => {
            // Create label
            const label = document.createElement('label');
            
            // Only set htmlFor for input types that have a single matching id
            // Don't set it for radio buttons since they have multiple inputs with different ids
            if (attr.type !== 'radio') {
                label.htmlFor = attr.id;
            }
            
            label.textContent = attr.id + ':';
            labelsContainer.appendChild(label);

            // Create input wrapper
            const wrapper = document.createElement('div');
            wrapper.className = 'input-wrapper';

            // Create input based on type
            if (attr.type === 'radio' && (currentDisease === 'cancer' || attr.id === 'Number of Major Vessels') ) {
                const radioGroup = document.createElement('div');
                radioGroup.className = 'radio-group';
                radioGroup.id = attr.id;

                for (let i = attr.min; i <= attr.max; i++) {
                    const radioLabel = document.createElement('label');
                    radioLabel.className = 'radio-scale-item';

                    const radioInput = document.createElement('input');
                    radioInput.type = 'radio';
                    radioInput.name = attr.id;
                    radioInput.id = `${attr.id}_${i}`;
                    radioInput.value = i;

                    const circleSpan = document.createElement('span');
                    circleSpan.className = 'radio-scale-circle';
                    circleSpan.textContent = i;

                    // Prevent focus from causing the page/scroll container to jump.
                    // We focus the actual input without scrolling it into view.
                    circleSpan.addEventListener('click', (e) => {
                        // The span isn't a real form control, so we select the hidden radio ourselves.
                        // Also prevent the browser from doing a scroll-to-focus.
                        e.preventDefault();

                        if (!radioInput) return;

                        radioInput.checked = true;
                        // Trigger native listeners/validation updates
                        radioInput.dispatchEvent(new Event('change', { bubbles: true }));

                        if (typeof radioInput.focus === 'function') {
                            try {
                                radioInput.focus({ preventScroll: true });
                            } catch {
                                radioInput.focus();
                            }
                        }
                    });

                    radioLabel.appendChild(radioInput);
                    radioLabel.appendChild(circleSpan);
                    radioGroup.appendChild(radioLabel);
                }
                wrapper.appendChild(radioGroup);

            } else if (attr.type === 'radio') {
                const radioGroup = document.createElement('div');
                radioGroup.className = 'radio-group';
                
                attr.options.forEach((option, index) => {
                    const radioOption = document.createElement('div');
                    radioOption.className = 'radio-option';
                    
                    const input = document.createElement('input');
                    input.type = 'radio';
                    input.name = attr.id;
                    input.id = `${attr.id}_${index}`;
                    input.value = option.value;
                    
                    const radioLabel = document.createElement('label');
                    radioLabel.htmlFor = `${attr.id}_${index}`;
                    radioLabel.textContent = option.label;
                    
                    radioOption.appendChild(input);
                    radioOption.appendChild(radioLabel);
                    radioGroup.appendChild(radioOption);
                });
                
                wrapper.appendChild(radioGroup);

            } else if (attr.type === 'dropdown') {
                const select = document.createElement('select');
                select.id = attr.id;
                select.className = 'dropdown-input';
                
                // Add default disabled option
                const defaultOption = document.createElement('option');
                defaultOption.value = '';
                defaultOption.textContent = 'Select an option';
                defaultOption.disabled = true;
                defaultOption.selected = true;
                select.appendChild(defaultOption);
                
                // Add attribute options
                attr.options.forEach(option => {
                    const optionElement = document.createElement('option');
                    optionElement.value = option.value;
                    optionElement.textContent = option.label;
                    select.appendChild(optionElement);
                });
                
                wrapper.appendChild(select);

            } else {
                // Regular number input
                const input = document.createElement('input');
                input.type = attr.type;
                input.id = attr.id;
                input.placeholder = attr.placeholder || '';
                input.min = attr.min;
                input.max = attr.max;
                input.step = attr.step || '1';
                // input.required = true;
                
                wrapper.appendChild(input);
            }

            inputsContainer.appendChild(wrapper);
        });

        // Initialize tooltips after DOM update
        document.querySelectorAll('.labels-container label').forEach((label, index) => {
            const attr = config.attributes[index];
            if (attr && attr.info) {
                label.setAttribute('data-tippy-content', attr.info.replace(/\n/g, '<br>'));
            }
        });

        // Initialize tippy
        tippy('[data-tippy-content]', {
            placement: 'top',
            theme: 'light',
        });

        // Re-attach event listener for info icon after form update
        const infoIcon = document.getElementById('infoIcon');
        if (infoIcon) {
            // Remove any existing listeners by cloning and replacing
            const newInfoIcon = infoIcon.cloneNode(true);
            infoIcon.parentNode.replaceChild(newInfoIcon, infoIcon);
            
            // Add new event listener
            newInfoIcon.addEventListener('click', function() {
                showDiseaseInfoModal(currentDisease);
            });
        }

        // Setup real-time validation listeners - ADD DEBUG LOG
        console.log('Setting up validation listeners...');
        if (window.singlePrediction && window.singlePrediction.setupRealTimeValidation) {
            console.log('setupRealTimeValidation function found, calling it...');
            window.singlePrediction.setupRealTimeValidation();
        } else {
            console.error('setupRealTimeValidation function NOT found!');
        }

        attachInlineRangeHelper(config);
    }

    function showDiseaseInfoModal(disease) {
        const info = diseaseInfoContent[disease];
        
        Swal.fire({
            html: `
                <div class="info-modal-content">
                    <h2>${info.title}</h2>
                    <div>${info.content}</div>
                </div>
            `,
            confirmButtonText: 'I understand',
            confirmButtonColor: info.buttonColor,
            customClass: {
                popup: 'info-modal',
                confirmButton: 'info-modal-button'
            },
            showCloseButton: false,
            focusConfirm: false
        });
    }

    
    // =====================================================================
    // BATCH PREDICTION MODE
    // =====================================================================

     function initializeBatchPrediction() {
        const uploadBtn = document.getElementById('uploadBtn');
        const checkValuesBtn = document.getElementById('checkValuesBtn');
        const downloadSampleCSVBtn = document.getElementById('downloadSampleCSV');
        const downloadBtn = document.getElementById('downloadBtn');
        const predictBtn = document.getElementById('predictBtn');
        const fileInput = document.getElementById('fileInput');
        const tableHeader = document.getElementById('tableHeader');
        const tableBody = document.getElementById('tableBody');

        // Upload button
        uploadBtn.addEventListener('click', () => {
            resetFileInput();
            fileInput.click();
        });

        // File input change
        fileInput.addEventListener('change', handleFileUpload);

        // Check values button
        checkValuesBtn.addEventListener('click', showValidAttributeValues);

        // Download sample CSV button
        downloadSampleCSVBtn.addEventListener('click', () => {
            const link = document.createElement('a');
            link.href = `../csv-files/diabetes_sample.csv`;
            link.download = `${currentDisease}_sample.csv`;
            link.click();
        });

        // Predict button
        predictBtn.addEventListener('click', handlePrediction);

        // Download button
        downloadBtn.addEventListener('click', handleDownload);
    }

    function updateBatchPredictionTable(config) {
        const tableHeader = document.getElementById('tableHeader');
        const tableBody = document.getElementById('tableBody');
        const tableContainer = document.querySelector('.table-container');
        
        tableHeader.innerHTML = '';
        
        // Add prediction column header (initially hidden)
        const predictionHeader = document.createElement('th');
        predictionHeader.textContent = 'Prediction';
        predictionHeader.className = 'prediction-column';
        predictionHeader.style.backgroundColor = getThemeColor();
        predictionHeader.style.color = 'white';
        predictionHeader.style.display = 'none';
        tableHeader.appendChild(predictionHeader);
        
        // Add probability column header (initially hidden)
        const probabilityHeader = document.createElement('th');
        probabilityHeader.textContent = 'Class Probability';
        probabilityHeader.className = 'prediction-column probability-column';
        probabilityHeader.style.backgroundColor = getThemeColor();
        probabilityHeader.style.color = 'white';
        probabilityHeader.style.display = 'none';
        tableHeader.appendChild(probabilityHeader);
        
        // Add attribute headers
        config.attributes.forEach(attr => {
            const th = document.createElement('th');
            th.textContent = attr.id;
            tableHeader.appendChild(th);
        });

        // Empty state: hide scrollbar chrome (still allows scrolling if needed)
        tableContainer?.classList.add('hide-scrollbar');
            
        // Clear table body and show empty placeholder rows with message
        const numColumns = config.attributes.length;
        // Make the placeholder height depend on the number of attributes per disease.
        // More columns => fewer rows, so the table doesn't overflow vertically.
        const numEmptyRows = Math.max(6, Math.min(10, Math.round(110 / Math.max(1, numColumns)) + 2));
        // Keep the message on a white stripe (even index)
        const messageRowIndex = Math.min(numEmptyRows - 1, 4 - (4 % 2));
        
        let emptyRowsHTML = '';
        for (let i = 0; i < numEmptyRows; i++) {
            const rowClass = i % 2 === 1 ? 'empty-row even-row' : 'empty-row';
            if (i === messageRowIndex) {
                // This row shows the message (on a white row)
                emptyRowsHTML += `<tr class="${rowClass}"><td colspan="${numColumns}" class="empty-table-message">No data uploaded yet. Upload a CSV file to see data here.</td></tr>`;
            } else {
                // Empty rows with just empty cells to show the striping
                emptyRowsHTML += `<tr class="${rowClass}">`;
                for (let j = 0; j < numColumns; j++) {
                    emptyRowsHTML += '<td>&nbsp;</td>';
                }
                emptyRowsHTML += '</tr>';
            }
        }
        tableBody.innerHTML = emptyRowsHTML;
    }

    async function handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
    
        console.log('File selected:', file.name, 'Type:', file.type, 'Size:', file.size);
    
        const predictedData = diseaseStates[currentDisease].batchPredictedData;
        
        if (predictedData) {
            // Show confirmation modal
            const result = await Swal.fire({
                icon: 'warning',
                title: 'Replace Current Data?',
                html: `
                    <p style="font-size: 16px; line-height: 1.6;">
                        You already have predictions displayed in the table.
                    </p>
                    <p style="font-size: 16px; line-height: 1.6; margin-top: 10px;">
                        <strong>Uploading a new file will:</strong>
                    </p>
                    <ul style="text-align: left; font-size: 15px; margin-top: 10px; padding-left: 30px;">
                        <li>Replace the currently uploaded data</li>
                        <li>Remove all prediction results from the table</li>
                        <li>Require you to make new predictions</li>
                    </ul>
                    <p style="font-size: 16px; margin-top: 15px; font-weight: 600;">
                        Do you want to continue?
                    </p>
                `,
                showCancelButton: true,
                confirmButtonColor: getThemeColor(),
                cancelButtonColor: '#6c757d',
                confirmButtonText: 'Yes, Upload New File',
                cancelButtonText: 'Cancel',
                width: '500px'
            });
            
            // If user cancels, reset file input and stop immediately
            if (!result.isConfirmed) {
                resetFileInput();
                return; 
            }
        }
    
        // User confirmed (or no predictions exist), NOW proceed with file processing
        const reader = new FileReader();
        reader.onload = async function(e) {
            try {
                const csvData = e.target.result;
                const parsedData = parseCSV(csvData);
                
                const isValid = await validateData(parsedData);
                
                if (isValid) {
                    diseaseStates[currentDisease].batchUploadedData = parsedData;
                    
                    // Clear predicted data since we have new upload
                    diseaseStates[currentDisease].batchPredictedData = null;
                    
                    displayData(parsedData, false);
                    
                    predictBtn.style.display = 'block';
                    downloadBtn.style.display = 'none';
                    
                    // Update button text after successful upload
                    updateUploadButtonText();
                    
                    Swal.fire({
                        icon: 'success',
                        title: 'Data Uploaded Successfully!',
                        text: `${parsedData.data.length} valid records loaded.`,
                        confirmButtonColor: getThemeColor()
                    });
                } else {
                    resetFileInput();
                }
            } catch (error) {
                console.error('CSV parsing error:', error);
                Swal.fire({
                    icon: 'error',
                    title: 'Invalid CSV Format',
                    text: error.message || 'Unable to parse the CSV file. Please check the format.',
                    confirmButtonColor: getThemeColor()
                });
                resetFileInput();
            }
        };
        
        reader.onerror = function() {
            Swal.fire({
                icon: 'error',
                title: 'File Reading Error',
                text: 'Unable to read the selected file. Please try again.',
                confirmButtonColor: getThemeColor()
            });
            resetFileInput();
        };
        
        reader.readAsText(file);
    }

    function parseCSV(csvData) {
        if (!csvData || typeof csvData !== 'string') {
            throw new Error('Invalid CSV data');
        }

        const lines = csvData.trim().split('\n');
        
        if (lines.length < 2) {
            throw new Error('CSV must have at least a header row and one data row');
        }

        const headerLine = lines[0];
        const headers = parseCSVLine(headerLine);
        
        if (headers.length === 0) {
            throw new Error('No headers found in CSV');
        }

        console.log('Parsed headers:', headers);

        const data = [];
        
        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line === '') continue;
            
            const values = parseCSVLine(line);
            
            if (values.length !== headers.length) {
                while (values.length < headers.length) {
                    values.push('');
                }
            }
            
            const row = {};
            headers.forEach((header, index) => {
                row[header] = values[index];
            });
            data.push(row);
        }

        if (data.length === 0) {
            throw new Error('No data rows found in CSV');
        }

        return { headers, data };
    }

    function parseCSVLine(line) {
        const result = [];
        let current = '';
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            
            if (char === '"') {
                inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
                result.push(current.trim());
                current = '';
            } else {
                current += char;
            }
        }
        
        result.push(current.trim());
        return result;
    }

    async function validateData(parsedData) {
        const config = diseaseConfigs[currentDisease];
        
        console.log('Expected headers:', config.attributes.map(attr => attr.id));
        console.log('CSV headers:', parsedData.headers);

        const expectedHeaders = config.attributes.map(attr => attr.id);
        const targetColumns = ['target', 'outcome', 'class', 'label', 'result'];
        
        const csvHeadersFiltered = parsedData.headers.filter(header => 
            !targetColumns.includes(header.toLowerCase())
        );
        
        console.log('CSV headers (filtered):', csvHeadersFiltered);

        const missingHeaders = [];
        const headerMap = {};
        
        expectedHeaders.forEach(expectedHeader => {
            let found = false;
            
            if (csvHeadersFiltered.includes(expectedHeader)) {
                headerMap[expectedHeader] = expectedHeader;
                found = true;
            } else {
                const lowerExpected = expectedHeader.toLowerCase();
                for (const csvHeader of csvHeadersFiltered) {
                    if (csvHeader.toLowerCase() === lowerExpected) {
                        headerMap[expectedHeader] = csvHeader;
                        found = true;
                        break;
                    }
                }
            }
            
            if (!found) {
                missingHeaders.push(expectedHeader);
            }
        });

        if (missingHeaders.length > 0) {
            console.log('Missing headers:', missingHeaders);
            
            Swal.fire({
                icon: 'error',
                title: 'Missing Required Headers',
                html: `
                    <div style="text-align: left;">
                        <p><strong>Missing headers:</strong></p>
                        <p style="font-size: 12px; color: #666;">${missingHeaders.join(', ')}</p>
                        <p><strong>Expected headers:</strong></p>
                        <p style="font-size: 12px; color: #666;">${expectedHeaders.join(', ')}</p>
                        <p><strong>Found headers:</strong></p>
                        <p style="font-size: 12px; color: #666;">${csvHeadersFiltered.join(', ')}</p>
                    </div>
                `,
                confirmButtonColor: getThemeColor()
            });
            
            return false;
        }

        // Validate data values
        const validationErrors = [];
        const invalidRowIndexes = new Set();
        
        for (let rowIndex = 0; rowIndex < parsedData.data.length; rowIndex++) {
            const row = parsedData.data[rowIndex];
            let hasErrors = false;
            
            for (const attr of config.attributes) {
                const value = row[headerMap[attr.id] || attr.id];
                
                if (!validateAttributeValue(value, attr)) {
                    validationErrors.push({
                        row: rowIndex + 2,
                        column: attr.label,
                        value: value || 'empty',
                        expected: getExpectedValueDescription(attr)
                    });
                    hasErrors = true;
                }
            }
            
            if (hasErrors) {
                invalidRowIndexes.add(rowIndex);
            }
        }

        if (validationErrors.length > 0) {
            console.log('Validation errors:', validationErrors);
            return await showValidationErrorsWithOption(validationErrors, invalidRowIndexes, parsedData);
        }

        return true;
    }

    async function showValidationErrorsWithOption(validationErrors, invalidRowIndexes, parsedData) {
        const totalRows = parsedData.data.length;
        const invalidCount = invalidRowIndexes.size;
        const validCount = totalRows - invalidCount;
        
        let errorMessage = '<div style="text-align: left; max-height: 300px; overflow-y: auto;">';
        errorMessage += `<p><strong>Data validation issues found:</strong></p>`;
        errorMessage += `<p style="margin-bottom: 15px; color: #666; font-size: 14px;">
            Total records: ${totalRows}<br>
            Invalid records: ${invalidCount}<br>
            Valid records: ${validCount}
        </p>`;
        
        if (validCount === 0) {
            errorMessage += '<p style="color: #d32f2f;"><strong>No valid records found. Upload cannot proceed.</strong></p>';
            errorMessage += '</div>';
            
            await Swal.fire({
                icon: 'error',
                title: 'No Valid Records',
                html: errorMessage,
                confirmButtonColor: getThemeColor(),
                confirmButtonText: 'OK'
            });
            
            return false;
        }
        
        errorMessage += '<p><strong>Sample validation errors:</strong></p>';
        errorMessage += '<ul style="font-size: 12px; margin-bottom: 15px;">';
        validationErrors.slice(0, 8).forEach(error => {
            errorMessage += `<li>Row ${error.row}, Column "${error.column}": Found "${error.value}", Expected ${error.expected}</li>`;
        });
        if (validationErrors.length > 8) {
            errorMessage += `<li><em>...and ${validationErrors.length - 8} more errors</em></li>`;
        }
        errorMessage += '</ul>';
        
        errorMessage += '<p style="font-weight: bold;">Do you want to proceed by removing the invalid records?</p>';
        errorMessage += '</div>';
        
        const result = await Swal.fire({
            icon: 'warning',
            title: 'Data Validation Issues Found',
            html: errorMessage,
            showCancelButton: true,
            confirmButtonColor: getThemeColor(),
            cancelButtonColor: '#d33',
            confirmButtonText: `Yes, Remove ${invalidCount} Invalid Records`,
            cancelButtonText: 'Cancel Upload',
            width: '600px'
        });
        
        if (result.isConfirmed) {
            // Filter out invalid rows
            const validData = parsedData.data.filter((row, index) => !invalidRowIndexes.has(index));
            parsedData.data = validData;
            
            return true; // Validation passed after cleaning
        } else {
            return false;
        }
    }

    function validateAttributeValue(value, attr) {
        if (value === undefined || value === null || value === '') {
            return false;
        }

        const numValue = parseFloat(value);

        if (attr.type === 'number') {
            if (isNaN(numValue)) return false;
            if (attr.min !== undefined && numValue < attr.min) return false;
            if (attr.max !== undefined && numValue > attr.max) return false;
        } else if (attr.type === 'categorical' || attr.type === 'radio' || attr.type === 'dropdown') {
            const validValues = attr.values || (attr.options ? attr.options.map(opt => opt.value) : []);
            if (validValues.length > 0 && !validValues.includes(numValue.toString()) && !validValues.includes(numValue)) {
                return false;
            }
        }

        return true;
    }

    function getExpectedValueDescription(attr) {
        if (attr.type === 'number') {
            let desc = 'Number';
            if (attr.min !== undefined && attr.max !== undefined) {
                desc += ` (${attr.min}-${attr.max})`;
            } else if (attr.min !== undefined) {
                desc += ` (≥${attr.min})`;
            } else if (attr.max !== undefined) {
                desc += ` (≤${attr.max})`;
            }
            return desc;
        } else if (attr.type === 'categorical' || attr.type === 'radio' || attr.type === 'dropdown') {
            const validValues = attr.values || (attr.options ? attr.options.map(opt => opt.value) : []);
            return `One of: ${validValues.join(', ')}`;
        } 
        return 'Valid value';
    }

    function displayData(data, showPrediction = false) {
        const tableBody = document.getElementById('tableBody');
        const config = diseaseConfigs[currentDisease];
        
        tableBody.innerHTML = '';
        
        // Show/hide prediction columns
        document.querySelectorAll('.prediction-column').forEach(col => {
            col.style.display = showPrediction ? 'table-cell' : 'none';
        });
        
        // Get predicted data from disease state
        const predictedData = diseaseStates[currentDisease].batchPredictedData;
        
        data.data.forEach((row, index) => {
            const tr = document.createElement('tr');
            
            // Add prediction cell if available
            if (showPrediction && predictedData) {
                const predictionCell = document.createElement('td');
                const prediction = predictedData.predictions[index];
                predictionCell.textContent = prediction === 1 ? config.positiveClass : config.negativeClass;
                predictionCell.className = 'prediction-column';
                tr.appendChild(predictionCell);
                
                // Add probability cell
                const probabilityCell = document.createElement('td');
                const probability = predictedData.probabilities[index];
                probabilityCell.textContent = `${probability}%`;
                probabilityCell.className = 'prediction-column probability-column';
                tr.appendChild(probabilityCell);
            } else if (showPrediction) {
                const predictionCell = document.createElement('td');
                predictionCell.className = 'prediction-column';
                tr.appendChild(predictionCell);
                
                const probabilityCell = document.createElement('td');
                probabilityCell.className = 'prediction-column probability-column';
                tr.appendChild(probabilityCell);
            }
            
            config.attributes.forEach(attr => {
                const td = document.createElement('td');
                td.textContent = row[attr.id] || '';
                tr.appendChild(td);
            });
            
            tableBody.appendChild(tr);
        });
        
        if (data.data.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="${config.attributes.length + 2}" class="empty-table-message">No valid data found.</td></tr>`;
        }
    }

    function showValidAttributeValues() {
        const config = diseaseConfigs[currentDisease];
        let content = `<div style="text-align: left; max-height: 400px; overflow-y: auto;">`;
        content += `<h3 style="margin-bottom: 15px; color: ${getThemeColor()};">Valid Attribute Values for ${config.name}</h3>`;
        
        config.attributes.forEach(attr => {
            content += `<div style="margin-bottom: 10px;">`;
            content += `<strong>${attr.id}:</strong> `;
            
            if (attr.type === 'number') {
                if (attr.min !== undefined && attr.max !== undefined) {
                    content += `${attr.min} to ${attr.max}`;
                } else if (attr.min !== undefined) {
                    content += `≥ ${attr.min}`;
                } else if (attr.max !== undefined) {
                    content += `≤ ${attr.max}`;
                } else {
                    content += `Any number`;
                }
            } else if (attr.type === 'categorical' || attr.type === 'radio' || attr.type === 'dropdown') {
                const validValues = attr.values || (attr.options ? attr.options.map(opt => `${opt.value} (${opt.label})`).join(', ') : []);
                content += Array.isArray(validValues) ? validValues.join(', ') : validValues;
            }
            
            content += `</div>`;
        });
        
        content += `</div>`;
        
        Swal.fire({
            title: 'Valid Attribute Values',
            html: content,
            width: '600px',
            confirmButtonColor: getThemeColor(),
            confirmButtonText: 'Close'
        });
    }

    async function handlePrediction() {
        const uploadedData = diseaseStates[currentDisease].batchUploadedData;
        if (!uploadedData) return;

        try {
            const predictBtn = document.getElementById('predictBtn');
            predictBtn.textContent = 'Processing...';
            predictBtn.disabled = true;

            const config = diseaseConfigs[currentDisease];
            
            const apiData = uploadedData.data.map(row => {
                const dataRow = {};
                config.attributes.forEach(attr => {
                    dataRow[attr.id] = row[attr.id];
                });
                return dataRow;
            });

            const response = await fetch(`http://127.0.0.1:5000${config.batchEndpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ data: apiData })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const result = await response.json();

            const predictedData = {
                predictions: result.predictions,
                probabilities: result.probabilities 
            };

            diseaseStates[currentDisease].batchPredictedData = predictedData;

            displayData(uploadedData, true);
            
            predictBtn.style.display = 'none';
            document.getElementById('downloadBtn').style.display = 'block';
            
            Swal.fire({
                icon: 'success',
                title: 'Prediction Complete!',
                text: 'The classification results have been added to the table.',
                confirmButtonColor: getThemeColor()
            });

        } catch (error) {
            console.error('Prediction error:', error);
            Swal.fire({
                icon: 'error',
                title: 'Prediction Failed',
                text: 'Unable to process the prediction. Please try again.',
                confirmButtonColor: getThemeColor()
            });
        } finally {
            const predictBtn = document.getElementById('predictBtn');
            predictBtn.textContent = 'Predict Class';
            predictBtn.disabled = false;
        }
    }

    function handleDownload() {
        const uploadedData = diseaseStates[currentDisease].batchUploadedData;
        const predictedData = diseaseStates[currentDisease].batchPredictedData;
        
        if (!uploadedData || !predictedData) {
            Swal.fire({
                icon: 'warning',
                title: 'No Predictions',
                text: 'Please make predictions first.',
                confirmButtonColor: getThemeColor()
            });
            return;
        }
    
        // Create CSV content
        const config = diseaseConfigs[currentDisease];
        const headers = ['Prediction', 'Class Probability', ...config.attributes.map(attr => attr.label)];
        let csvContent = headers.join(',') + '\n';
    
        uploadedData.data.forEach((row, index) => {
            const prediction = predictedData.predictions[index] === 1 ? config.positiveClass : config.negativeClass;
            const probability = `${predictedData.probabilities[index]}%`;
            const rowData = [prediction, probability, ...config.attributes.map(attr => row[attr.id])];
            csvContent += rowData.join(',') + '\n';
        });

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${currentDisease}_predictions_cleaned.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        Swal.fire({
            icon: 'success',
            title: 'Download Complete!',
            text: 'The result data with predictions has been downloaded.',
            confirmButtonColor: getThemeColor(),
            timer: 2000,
            showConfirmButton: false
        });
    }

    function updateUploadButtonText() {
        const uploadBtn = document.getElementById('uploadBtn');
        const uploadedData = diseaseStates[currentDisease].batchUploadedData;
        
        if (uploadedData && uploadedData.data && uploadedData.data.length > 0) {
            uploadBtn.textContent = 'Upload New Data';
        } else {
            uploadBtn.textContent = 'Upload Data';
        }
    }

    function resetFileInput() {
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.value = '';
        }
    }

    function getThemeColor() {
        const colors = {
            diabetes: '#096F29',
            heart: '#811111',
            cancer: '#1E2F4E'
        };
        return colors[currentDisease];
    }
    

    initializePage();
});