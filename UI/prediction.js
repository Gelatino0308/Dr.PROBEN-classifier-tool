document.addEventListener('DOMContentLoaded', () => {
    let currentDisease = 'diabetes';
    let currentMode = 'single'; // 'single' or 'batch'
    
    // State management for both modes
    const diseaseStates = {
        diabetes: { 
            singleFormData: {}, 
            singleResultData: null,
            batchUploadedData: null,
            batchPredictedData: null
        },
        heart: { 
            singleFormData: {}, 
            singleResultData: null,
            batchUploadedData: null,
            batchPredictedData: null
        },
        cancer: { 
            singleFormData: {}, 
            singleResultData: null,
            batchUploadedData: null,
            batchPredictedData: null
        }
    };

    // Disease configurations
    const diseaseConfigs = {
        diabetes: {
            name: 'Diabetes',
            endpoint: '/api/predict/diabetes',
            batchEndpoint: '/api/predict/diabetes/batch',
            positiveClass: 'DIABETIC',
            negativeClass: 'NON-DIABETIC',
            positiveDesc: "Diabetic means the person has diabetes, a chronic disease that affects how your body turns food into energy. It occurs when your pancreas doesn't make enough insulin or your cells don't respond to insulin properly.",
            negativeDesc: 'Non-diabetic means the absence of diabetes. Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces.',
            attributes: [
                { id: 'pregnancies', label: 'Number of Pregnancies', placeholder: '0', min: '0', type: 'number', 
                    info: 'If you have been pregnant twice, you would enter "2." If you have never been pregnant, you would enter "0."' 
                },
                { id: 'plasma', label: 'Plasma Glucose Concentration', placeholder: '0 (mg/dL)', min: '0', type: 'number', 
                    info: 'This measures the amount of sugar in your blood. You will need to get this value from a recent blood test, often called a blood sugar test or glucose test.' 
                },
                { id: 'BP', label: 'Diastolic Blood Pressure', placeholder: '0 (mm Hg)', min: '0', type: 'number',
                    info: 'This is the second, or lower, number in a blood pressure reading. A reading is typically written as two numbers, like "120/80."'
                },
                { id: 'skin', label: 'Triceps Skin Fold Thickness', placeholder: '0 (mm)', min: '0', type: 'number',
                    info: 'This value is a way to estimate the amount of body fat by measuring the thickness of a fold of skin and fat on the back of your upper arm.'    
                },
                { id: 'insulin', label: '2-Hour Serum Insulin', placeholder: '0 (µU/mL)', min: '0', type: 'number',
                    info: 'This measures the amount of insulin in your blood specifically two hours after you\'ve taken a glucose tolerance test.'
                },
                { id: 'BMI', label: 'Body Mass Index', placeholder: '0.0 (kg/m²)', min: '0', type: 'number', step: 'any',
                    info: 'Your BMI is a value calculated from your weight and height that helps determine if you are at a healthy weight.'
                },
                { id: 'pedigree', label: 'Diabetes Pedigree Function', placeholder: '0.000', min: '0', type: 'number', step: 'any',
                    info: 'This is a complex score that quantifies the genetic risk of diabetes based on your family history.'
                },
                { id: 'age', label: 'Age', placeholder: '0', min: '0', type: 'number',
                    info:'This is your current age.'
                }
            ]
        },
        heart: {
            name: 'Heart Disease',
            endpoint: '/api/predict/heart',
            batchEndpoint: '/api/predict/heart/batch',
            positiveClass: 'POSITIVE',
            negativeClass: 'NEGATIVE',
            positiveDesc: "Positive means the presence of heart disease. Heart disease refers to several types of heart conditions that affect the heart's ability to function normally.",
            negativeDesc: 'Negative means the absence of cardiovascular conditions. A healthy heart efficiently pumps blood throughout the body.',
            attributes: [
                { id: 'age', label: 'Age', placeholder: '0', min: '0', type: 'number', info: 'This is your current age.' },
                { id: 'sex', label: 'Sex', type: 'radio', options: [{ value: '1', label: 'Male' }, { value: '0', label: 'Female' }], info: 'This refers to your biological sex.' },
                { id: 'cp', label: 'Chest Pain Type', type: 'dropdown', options: [{ value: '0', label: 'Asymptomatic' }, { value: '1', label: 'Atypical Angina' }, { value: '2', label: 'Non-anginal Pain' }, { value: '3', label: 'Typical Angina' }], info: 'Common types of chest pain.' },
                { id: 'trestbps', label: 'Resting Blood Pressure', placeholder: '0 (mm Hg)', min: '0', type: 'number', info: 'This is the top number of your blood pressure reading.' },
                { id: 'chol', label: 'Serum Cholesterol', placeholder: '0 (mg/dL)', min: '0', type: 'number', info: 'This is the amount of total cholesterol in your blood.' },
                { id: 'fbs', label: 'FBS > 120mg/dL', type: 'radio', options: [{ value: '1', label: 'True' }, { value: '0', label: 'False' }], info: 'Indicates whether fasting blood sugar is greater than 120 mg/dL.' },
                { id: 'restecg', label: 'Resting ECG Results', type: 'dropdown', options: [{ value: '0', label: 'Normal' }, { value: '1', label: 'ST-T Wave Abnormality' }, { value: '2', label: 'Left Ventricular Hypertrophy' }], info: 'Record of heart electrical activity at rest.' },
                { id: 'thalach', label: 'Maximum Heart Rate', placeholder: '0', min: '0', type: 'number', info:'Highest heart rate reached during stress test.' },
                { id: 'exang', label: 'Exercise Induced Angina', type: 'radio', options: [{ value: '1', label: 'Yes' }, { value: '0', label: 'No' }], info: 'Chest pain during exercise.' },
                { id: 'oldpeak', label: 'ST Depression (Oldpeak)', placeholder: '0.0', min: '0', type: 'number', step: 'any', info: 'ST segment depression during exercise.' },
                { id: 'slope', label: 'Slope of Peak Exercise ST', type: 'dropdown', options: [{ value: '0', label: 'Downsloping' }, { value: '1', label: 'Flat' }, { value: '2', label: 'Upsloping' }], info:'Slope of ST segment on ECG.' },
                { id: 'ca', label: 'Number of Major Vessels', placeholder: '0-4', min: '0', max: '4', type: 'slider', default: '0', info: 'Number of major blood vessels narrowed.' },
                { id: 'thal', label: 'Thalassemia', type: 'dropdown', options: [{ value: '1', label: 'Normal' }, { value: '2', label: 'Fixed Defect' }, { value: '3', label: 'Reversible Defect' }], info: 'Thallium scan results.' }
            ]
        },
        cancer: {
            name: 'Breast Cancer',
            endpoint: '/api/predict/cancer',
            batchEndpoint: '/api/predict/cancer/batch',
            positiveClass: 'MALIGNANT',
            negativeClass: 'BENIGN',
            positiveDesc: "Malignant means the tumor is cancerous and can spread to other parts of the body.",
            negativeDesc: 'Benign means the tumor is non-cancerous and does not spread to other parts of the body.',
            attributes: [
                { id: 'clump_thickness', label: 'Clump Thickness', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '0', info:'Degree of cell clustering.' },
                { id: 'uniformity_cell_size', label: 'Uniformity of Cell Size', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '0', info:'Consistency in cell sizes.' },
                { id: 'uniformity_cell_shape', label: 'Uniformity of Cell Shape', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '0', info:'Uniformity of cell shapes.' },
                { id: 'marginal_adhesion', label: 'Marginal Adhesion', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '0', info:'Extent of cell adhesion.' },
                { id: 'single_epithelial_cell_size', label: 'Single Epithelial Cell Size', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '0', info:'Average epithelial cell size.' },
                { id: 'bare_nuclei', label: 'Bare Nuclei', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '0', info:'Presence of nuclei without cytoplasm.' },
                { id: 'bland_chromatin', label: 'Bland Chromatin', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '0', info:'Chromatin texture and appearance.' },
                { id: 'normal_nucleoli', label: 'Normal Nucleoli', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '0', info:'Nucleoli visibility and prominence.' },
                { id: 'mitoses', label: 'Mitoses', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '0', info:'Frequency of cell division.' }
            ]
        }
    };

    // Initialize page
    function initializePage() {
        const storedDisease = sessionStorage.getItem('selectedDisease');
        if (storedDisease && diseaseConfigs[storedDisease]) {
            currentDisease = storedDisease;
            sessionStorage.removeItem('selectedDisease');
        }

        updatePageContent(currentDisease);
        setupEventListeners();
        initializeSinglePrediction();
        initializeBatchPrediction();
    }

    // Setup event listeners
    function setupEventListeners() {
        // Segment control buttons
        document.querySelectorAll('.segment-button').forEach(button => {
            button.addEventListener('click', function() {
                const mode = this.dataset.mode;
                if (mode !== currentMode) {
                    switchMode(mode);
                }
            });
        });
    }

    // Switch between single and batch mode
    function switchMode(mode) {
        currentMode = mode;
        
        // Update segment control UI
        document.querySelectorAll('.segment-button').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
        
        // Show/hide sections
        document.getElementById('singlePredictionSection').classList.toggle('active', mode === 'single');
        document.getElementById('batchPredictionSection').classList.toggle('active', mode === 'batch');
    }

    // Update page content based on disease
    function updatePageContent(disease) {
        currentDisease = disease;
        const config = diseaseConfigs[disease];
        
        // Update title
        document.getElementById('diseaseTitle').textContent = config.name;
        
        // Apply theme
        document.body.className = `theme-${disease}`;
        
        // Update single prediction form
        updateSinglePredictionForm(config);
        
        // Update batch prediction table
        updateBatchPredictionTable(config);
    }

    // Back button functionality
    window.goBack = function() {
        window.location.href = 'index.html';
    };

    
    


    // Single Prediction Functionality
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
                backgroundColor: ['#ffffff', '#008042'],
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
                    animateScale: false
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        enabled: true,
                        callbacks: {
                            label: function(context) {
                                const dataIndex = context.dataIndex;
                                const value = context.parsed;
                                const config = diseaseConfigs[currentDisease];

                                if (dataIndex === 0) {
                                    return `${config.positiveClass}: ${value.toFixed(2)}%`;
                                } else {
                                    return `${config.negativeClass}: ${value.toFixed(2)}%`;
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

        // Update chart colors based on disease theme
        function updateChartColors(disease) {
            const themeColors = {
                diabetes: ['#ffffff', '#008042'],
                heart: ['#ffffff', '#651515'],
                cancer: ['#ffffff', '#073056']
            };

            chart.data.datasets[0].backgroundColor = themeColors[disease];
            chart.update('none');
        }

        // Handle form submission
        form.addEventListener('submit', async function(e) {
            e.preventDefault();

            const config = diseaseConfigs[currentDisease];

            // Special validation for cancer sliders
            if (currentDisease === 'cancer') {
                const unmodifiedFields = [];
                config.attributes.forEach(attr => {
                    const input = document.getElementById(attr.id);
                    if (input && input.type === 'range' && input.value === '0') {
                        unmodifiedFields.push(attr.label);
                    }
                });

                if (unmodifiedFields.length > 0) {
                    showValidationModal(unmodifiedFields);
                    return;
                }
            }

            // Get form data dynamically
            const formData = {};
            config.attributes.forEach(attr => {
                const element = document.getElementById(attr.id);
                if (element) {
                    if (attr.type === 'radio') {
                        const selected = document.querySelector(`input[name="${attr.id}"]:checked`);
                        formData[attr.id] = selected ? parseFloat(selected.value) : null;
                    } else {
                        formData[attr.id] = parseFloat(element.value);
                    }
                }
            });

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

                // Store result in state
                diseaseStates[currentDisease].singleResultData = result;

                // Display result
                displayResult.style.display = 'flex';
                const resultElement = displayResult.querySelector('h1');
                const descElement = displayResult.querySelector('p');

                if (result.prediction === 1) {
                    resultElement.textContent = config.positiveClass;
                    descElement.textContent = config.positiveDesc;
                } else {
                    resultElement.textContent = config.negativeClass;
                    descElement.textContent = config.negativeDesc;
                }

                // Update chart
                percentage = result.probability;
                percentText.textContent = `${percentage.toFixed(2)}%`;
                chart.data.datasets[0].data = [percentage, 100 - percentage];
                chart.update('none');

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
                config.attributes.forEach(attr => {
                    if (attr.type === 'slider') {
                        const input = document.getElementById(attr.id);
                        const valueDisplay = input?.nextElementSibling;
                        if (input && valueDisplay) {
                            input.value = attr.default || '0';
                            valueDisplay.textContent = input.value;
                            if (input.value === '0') {
                                valueDisplay.classList.add('slider-unmodified');
                            }
                        }
                    }
                });
            }, 0);

            displayResult.style.display = 'none';
            percentText.textContent = '--%';
            chart.data.datasets[0].data = [0, 100];
            chart.update('none');
        });

        // Expose functions for global access
        window.singlePrediction = {
            updateChartColors,
            chart
        };
    }

    function updateSinglePredictionForm(config) {
        const labelsContainer = document.querySelector('.labels-container');
        const inputsContainer = document.querySelector('.inputs-container');

        labelsContainer.innerHTML = '';
        inputsContainer.innerHTML = '';

        config.attributes.forEach(attr => {
            // Create label
            const label = document.createElement('label');
            label.htmlFor = attr.id;
            label.textContent = attr.label + ':';
            label.setAttribute('data-tippy-content', attr.info);
            labelsContainer.appendChild(label);

            // Create input wrapper
            const wrapper = document.createElement('div');
            wrapper.className = 'input-wrapper';

            // Create input based on type
            if (attr.type === 'radio') {
                const radioGroup = document.createElement('div');
                radioGroup.className = 'radio-group';
                attr.options.forEach(option => {
                    const radioOption = document.createElement('div');
                    radioOption.className = 'radio-option';
                    const input = document.createElement('input');
                    input.type = 'radio';
                    input.id = `${attr.id}_${option.value}`;
                    input.name = attr.id;
                    input.value = option.value;
                    input.required = true;
                    const radioLabel = document.createElement('label');
                    radioLabel.htmlFor = `${attr.id}_${option.value}`;
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
                select.required = true;
                attr.options.forEach(option => {
                    const opt = document.createElement('option');
                    opt.value = option.value;
                    opt.textContent = option.label;
                    select.appendChild(opt);
                });
                wrapper.appendChild(select);
            } else if (attr.type === 'slider') {
                const slider = document.createElement('input');
                slider.type = 'range';
                slider.id = attr.id;
                slider.className = 'slider-input';
                slider.min = attr.min;
                slider.max = attr.max;
                slider.value = attr.default || '0';
                slider.required = true;

                const valueDisplay = document.createElement('span');
                valueDisplay.className = 'slider-value';
                valueDisplay.textContent = slider.value;
                if (slider.value === '0') {
                    valueDisplay.classList.add('slider-unmodified');
                }

                slider.addEventListener('input', function() {
                    valueDisplay.textContent = this.value;
                    if (this.value !== '0') {
                        valueDisplay.classList.remove('slider-unmodified');
                    } else {
                        valueDisplay.classList.add('slider-unmodified');
                    }
                });

                wrapper.appendChild(slider);
                wrapper.appendChild(valueDisplay);
            } else {
                const input = document.createElement('input');
                input.type = attr.type;
                input.id = attr.id;
                input.placeholder = attr.placeholder;
                input.min = attr.min;
                input.max = attr.max;
                input.step = attr.step || '';
                input.required = true;
                wrapper.appendChild(input);
            }

            inputsContainer.appendChild(wrapper);
        });

        // Initialize tooltips
        tippy('[data-tippy-content]', {
            theme: 'light',
            placement: 'right',
            arrow: true,
            maxWidth: 300
        });
    }

    function showValidationModal(unmodifiedFields) {
        const fieldsList = unmodifiedFields.map(field => `• ${field}`).join('<br>');

        Swal.fire({
            icon: 'warning',
            title: 'Invalid Form Values',
            html: `
                <p style="margin-bottom: 10px; line-height: 2;">Each field only accepts input between <strong>1-10</strong>, 
                    so please adjust all slider values accordingly before submitting. To know why, click the 
                    <svg style="height: 25px;" class="info-icon" id="infoIcon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
                    </svg> 
                    button above the form.
                </p>
                <p style="margin-bottom: 10px; font-weight: 500; color: #555;">The following fields still have invalid values (0):</p>
                <div style="
                    background: #f8f9fa; 
                    border-left: 4px solid #e74c3c; 
                    padding: 16px; 
                    border-radius: 4px; 
                    text-align: left; 
                    margin-top: 12px;
                    max-height: 180px;
                    overflow-y: auto;
                ">
                    <div style="color: #e74c3c; font-weight: 500; line-height: 1.8;">
                        ${fieldsList}
                    </div>
                </div>
            `,
            confirmButtonText: 'Got it!',
            confirmButtonColor: '#073056',
            customClass: {
                popup: 'validation-modal-popup',
                confirmButton: 'validation-modal-button'
            }
        });
    }






     // Batch Prediction Functionality
     function initializeBatchPrediction() {
        const uploadBtn = document.getElementById('uploadBtn');
        const checkValuesBtn = document.getElementById('checkValuesBtn');
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

        // Predict button
        predictBtn.addEventListener('click', handlePrediction);

        // Download button
        downloadBtn.addEventListener('click', handleDownload);
    }

    function updateBatchPredictionTable(config) {
        const tableHeader = document.getElementById('tableHeader');
        const tableBody = document.getElementById('tableBody');
        
        tableHeader.innerHTML = '';
        
        // Add prediction column header (initially hidden)
        const predictionHeader = document.createElement('th');
        predictionHeader.textContent = 'Prediction';
        predictionHeader.className = 'prediction-column';
        predictionHeader.style.display = 'none';
        tableHeader.appendChild(predictionHeader);
        
        // Add attribute headers
        config.attributes.forEach(attr => {
            const th = document.createElement('th');
            th.textContent = attr.label;
            tableHeader.appendChild(th);
        });
        
        // Clear table body
        tableBody.innerHTML = `<tr><td colspan="${config.attributes.length + 1}" class="empty-table-message">No data uploaded yet. Upload a CSV file to see data here.</td></tr>`;
    }

    function handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        console.log('File selected:', file.name, 'Type:', file.type, 'Size:', file.size);

        // Reset current data
        diseaseStates[currentDisease].batchUploadedData = null;
        diseaseStates[currentDisease].batchPredictedData = null;
        
        // Reset UI state
        document.getElementById('predictBtn').style.display = 'none';
        document.getElementById('downloadBtn').style.display = 'none';
        document.querySelector('.prediction-column').style.display = 'none';

        const reader = new FileReader();
        reader.onload = async function(e) {
            try {
                const csvData = e.target.result;
                const parsedData = parseCSV(csvData);
                
                const isValid = await validateData(parsedData);
                
                if (isValid) {
                    if (!diseaseStates[currentDisease].batchUploadedData) {
                        diseaseStates[currentDisease].batchUploadedData = parsedData;
                        displayData(parsedData);
                        document.getElementById('predictBtn').style.display = 'block';
                        
                        await Swal.fire({
                            icon: 'success',
                            title: 'File Uploaded Successfully!',
                            text: `Loaded ${parsedData.data.length} records.`,
                            confirmButtonColor: getThemeColor()
                        });
                    }
                }
            } catch (error) {
                console.error('CSV parsing error:', error);
                Swal.fire({
                    icon: 'error',
                    title: 'Invalid CSV File',
                    text: error.message || 'Unable to parse the CSV file. Please check the format.',
                    confirmButtonColor: getThemeColor()
                });
            }
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
                const value = row[attr.id];
                if (!validateAttributeValue(value, attr)) {
                    validationErrors.push({
                        row: rowIndex + 2,
                        column: attr.label,
                        value: value,
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
                title: 'All Records Invalid',
                html: errorMessage,
                confirmButtonColor: getThemeColor()
            });
            
            return false;
        }
        
        errorMessage += '<p><strong>Sample validation errors:</strong></p>';
        errorMessage += '<ul style="font-size: 12px; margin-bottom: 15px;">';
        validationErrors.slice(0, 8).forEach(error => {
            errorMessage += `<li>Row ${error.row}, ${error.column}: "${error.value}" (Expected: ${error.expected})</li>`;
        });
        if (validationErrors.length > 8) {
            errorMessage += `<li>... and ${validationErrors.length - 8} more errors</li>`;
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
            const cleanedData = {
                headers: parsedData.headers,
                data: parsedData.data.filter((row, index) => !invalidRowIndexes.has(index))
            };
            
            diseaseStates[currentDisease].batchUploadedData = cleanedData;
            
            displayData(cleanedData);
            document.getElementById('predictBtn').style.display = 'block';
            document.getElementById('downloadBtn').style.display = 'none';
            
            await Swal.fire({
                icon: 'success',
                title: 'File Uploaded Successfully!',
                html: `
                    <div style="text-align: center;">
                        <p>Loaded ${cleanedData.data.length} valid records.</p>
                        <p style="color: #666; font-size: 14px;">
                            ${invalidCount} invalid records were removed.
                        </p>
                    </div>
                `,
                confirmButtonColor: getThemeColor()
            });
            
            return true;
        } else {
            diseaseStates[currentDisease].batchUploadedData = null;
            diseaseStates[currentDisease].batchPredictedData = null;
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
        } else if (attr.type === 'slider') {
            if (isNaN(numValue)) return false;
            if (attr.min !== undefined && numValue < parseFloat(attr.min)) return false;
            if (attr.max !== undefined && numValue > parseFloat(attr.max)) return false;
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
        } else if (attr.type === 'slider') {
            return `${attr.min}-${attr.max}`;
        }
        return 'Valid value';
    }

    function displayData(data, showPrediction = false) {
        const tableBody = document.getElementById('tableBody');
        const config = diseaseConfigs[currentDisease];
        
        tableBody.innerHTML = '';
        
        document.querySelector('.prediction-column').style.display = showPrediction ? 'table-cell' : 'none';
        
        data.data.forEach((row, index) => {
            const tr = document.createElement('tr');
            
            if (showPrediction && diseaseStates[currentDisease].batchPredictedData) {
                const predCell = document.createElement('td');
                predCell.textContent = diseaseStates[currentDisease].batchPredictedData[index];
                predCell.className = 'prediction-column';
                tr.appendChild(predCell);
            } else if (showPrediction) {
                const predCell = document.createElement('td');
                predCell.textContent = '-';
                predCell.className = 'prediction-column';
                tr.appendChild(predCell);
            }
            
            config.attributes.forEach(attr => {
                const td = document.createElement('td');
                td.textContent = row[attr.id] || '';
                tr.appendChild(td);
            });
            
            tableBody.appendChild(tr);
        });
        
        if (data.data.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="${config.attributes.length + 1}" class="empty-table-message">No valid data found.</td></tr>`;
        }
    }

    function showValidAttributeValues() {
        const config = diseaseConfigs[currentDisease];
        let content = `<div style="text-align: left; max-height: 400px; overflow-y: auto;">`;
        content += `<h3 style="margin-bottom: 15px; color: ${getThemeColor()};">Valid Attribute Values for ${config.name}</h3>`;
        
        config.attributes.forEach(attr => {
            content += `<div style="margin-bottom: 10px;">`;
            content += `<strong>${attr.label}:</strong> `;
            
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
            } else if (attr.type === 'slider') {
                content += `${attr.min} to ${attr.max}`;
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
                const dataPoint = {};
                config.attributes.forEach(attr => {
                    dataPoint[attr.id] = parseFloat(row[attr.id]);
                });
                return dataPoint;
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
            
            const predictedData = result.predictions.map(pred => 
                pred === 1 ? config.positiveClass : config.negativeClass
            );

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
        
        if (!uploadedData || !predictedData) return;

        const config = diseaseConfigs[currentDisease];
        const headers = ['Prediction', ...config.attributes.map(attr => attr.label)];
        let csvContent = headers.join(',') + '\n';

        uploadedData.data.forEach((row, index) => {
            const rowData = [
                predictedData[index],
                ...config.attributes.map(attr => row[attr.id])
            ];
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
            text: 'The cleaned data with predictions has been downloaded.',
            confirmButtonColor: getThemeColor(),
            timer: 2000,
            showConfirmButton: false
        });
    }

    function resetFileInput() {
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.value = '';
        }
    }

    function getThemeColor() {
        const colors = {
            diabetes: '#00BF63',
            heart: '#DF6565',
            cancer: '#0097B2'
        };
        return colors[currentDisease];
    }
    

    initializePage();
});