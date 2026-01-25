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
            negativeDesc: 'Non-diabetic means the absence of diabetes. Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood glucose.',
            attributes: [
                { id: 'Number of Pregnancies', label: 'Number of Pregnancies', placeholder: '0', min: '0', type: 'number', 
                    info: 'If you have been pregnant twice, you would enter "2." If you have never been pregnant, you would enter "0."' 
                },
                { id: 'Plasma Glucose Concentration', label: 'Plasma Glucose Concentration', placeholder: '0 (mg/dL)', min: '0', type: 'number', 
                    info: 'This measures the amount of sugar in your blood. You will need to get this value from a recent blood test, often called a blood sugar test or glucose test. Look for a result listed as "Fasting Plasma Glucose" or similar, which is measured in milligrams per deciliter (mg/dL).'
                },
                { id: 'Diastolic Blood Pressure', label: 'Diastolic Blood Pressure', placeholder: '0 (mm Hg)', min: '0', type: 'number',
                    info: 'This is the second, or lower, number in a blood pressure reading. A reading is typically written as two numbers, like "120/80." In this example, "80" is the diastolic pressure. You can get this from a recent doctor\'s visit or a home blood pressure monitor.'
                },
                { id: 'Triceps Skin Fold Thickness', label: 'Triceps Skin Fold Thickness', placeholder: '0 (mm)', min: '0', type: 'number',
                    info: 'This value is a way to estimate the amount of body fat by measuring the thickness of a fold of skin and fat on the back of your upper arm. This measurement is usually taken with a special tool called a caliper. You will need to get this value from your doctor.'    
                },
                { id: '2-Hour Serum Insulin', label: '2-Hour Serum Insulin', placeholder: '0 (µU/mL)', min: '0', type: 'number',
                    info: 'This measures the amount of insulin in your blood specifically two hours after you\'ve taken a glucose tolerance test. It shows how well your body processes sugar over time. This value should be obtained from a specific blood test.'
                },
                { id: 'Body Mass Index', label: 'Body Mass Index', placeholder: '0.0 (kg/m²)', min: '0', type: 'number', step: 'any',
                    info: 'Your BMI is a value calculated from your weight and height that helps determine if you are at a healthy weight. To find your BMI, you can use an online calculator. Simply enter your height and weight, and the calculator will provide your BMI value. For example, if you weigh 150 lbs and are 5\'5" tall, your BMI is approximately 25.'
                },
                { id: 'Diabetes Pedigree Function', label: 'Diabetes Pedigree Function', placeholder: '0.000', min: '0', type: 'number', step: 'any',
                    info: 'This is a complex score that quantifies the genetic risk of diabetes based on your family history. You won\'t have a number for this yourself. This value is typically calculated by the diagnostic tool based on the family history information you provide, such as whether your parents or siblings have diabetes.'
                },
                { id: 'Age', label: 'Age', placeholder: '0', min: '0', type: 'number',
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
            positiveDesc: "Positive means the presence of heart disease. Heart disease refers to several types of heart conditions that affect the heart's ability to function normally. It includes coronary artery disease, heart rhythm problems, and heart defects.",
            negativeDesc: 'Negative means the absence of cardiovascular conditions. A healthy heart efficiently pumps blood throughout the body, delivering oxygen and nutrients to organs and tissues.',
            attributes: [
                { id: 'Age', label: 'Age', placeholder: '0', min: '0', type: 'number',
                    info: 'This is your current age.'
                },
                { id: 'Sex', label: 'Sex', type: 'radio', 
                    options: [
                        { value: '1', label: 'Male' },
                        { value: '0', label: 'Female' }
                    ],
                    info: 'This refers to your biological sex.'
                },
                { id: 'Chest Pain Type', label: 'Chest Pain Type', type: 'dropdown', 
                    options: [
                        { value: '0', label: 'Asymptomatic' },
                        { value: '1', label: 'Atypical Angina' },
                        { value: '2', label: 'Non-anginal Pain' },
                        { value: '3', label: 'Typical Angina' }
                    ],
                    info: 'Common types are:\n• Typical Angina: Chest pain caused by reduced blood flow to the heart\n• Atypical Angina: Chest discomfort that doesn\'t follow typical angina patterns\n• Non-anginal Pain: Chest pain not related to heart conditions\n• Asymptomatic: No chest pain symptoms'
                },
                { id: 'Resting Blood Pressure', label: 'Resting Blood Pressure', placeholder: '0 (mm Hg)', min: '0', type: 'number',
                    info: 'This is the top number of your blood pressure reading, measured while you are at rest. It is measured in millimeters of mercury (mm Hg).'
                },
                { id: 'Serum Cholesterol', label: 'Serum Cholesterol', placeholder: '0 (mg/dL)', min: '0', type: 'number',
                    info: 'This is the amount of total cholesterol in your blood. It is measured in milligrams per deciliter (mg/dL).'
                },
                { id: 'FBS > 120mg/dL', label: 'FBS > 120mg/dL', type: 'radio', 
                    options: [
                        { value: '1', label: 'True' },
                        { value: '0', label: 'False' }
                    ],
                    info: 'This indicates whether your fasting blood sugar is greater than 120 mg/dL. This is a common threshold for diagnosing prediabetes or diabetes.\n• True: Your fasting blood sugar is greater than 120 mg/dL.\n• False: Your fasting blood sugar is 120 mg/dL or less.'
                },
                { id: 'Resting ECG Results', label: 'Resting ECG Results', type: 'dropdown', 
                    options: [
                        { value: '0', label: 'Normal' },
                        { value: '1', label: 'ST-T Wave Abnormality' },
                        { value: '2', label: 'Left Ventricular Hypertrophy' }
                    ],
                    info: 'This is a record of your heart\'s electrical activity while you are at rest. You will need a recent ECG report.\n• Normal: No significant abnormalities.\n• ST-T Wave Abnormality: Minor changes that could indicate a heart issue.\n• Left Ventricular Hypertrophy (LVH): Thickening of the heart\'s main pumping chamber.'
                },
                { id: 'Maximum Heart Rate', label: 'Maximum Heart Rate', placeholder: '0', min: '0', type: 'number',
                    info:'This is the highest heart rate you reached during a stress or exercise test. This measurement is often taken on a treadmill or stationary bike while your heart rate is monitored.'
                },
                { id: 'Exercise Induced Angina', label: 'Exercise Induced Angina', type: 'radio', 
                    options: [
                        { value: '1', label: 'Yes' },
                        { value: '0', label: 'No' }
                    ],
                    info: 'This indicates whether you experienced chest pain during physical exercise.\n• Yes: You experienced chest pain during exercise.\n• No: You did not experience chest pain during exercise.'
                },
                { id: 'ST Depression (Oldpeak)', label: 'ST Depression (Oldpeak)', placeholder: '0.0', min: '0', type: 'number', step: 'any',
                    info: 'This measures the amount of depression in the ST segment of your ECG during exercise, which can be a sign of reduced blood flow to the heart. The value is measured in millimeters.'
                },
                { id: 'Slope of Peak Exercise ST', label: 'Slope of Peak Exercise ST', type: 'dropdown', 
                    options: [
                        { value: '0', label: 'Downsloping' },
                        { value: '1', label: 'Flat' },
                        { value: '2', label: 'Upsloping' }
                    ],
                    info:'Describes the slope of ST segment on your ECG during an exercise stress test.\n• Upsloping: The ST segment goes up.\n• Flat: The ST segment is horizontal.\n• Downsloping: The ST segment goes down. A downsloping or flat slope can be a sign of heart disease.'
                },
                { id: 'Number of Major Vessels', label: 'Number of Major Vessels', placeholder: '0-3', min: '0', max: '3', type: 'slider', default: '0',
                    info: 'This refers to the number of major blood vessels (0 to 3) that are significantly narrowed as seen in a coronary angiography. This value is provided by a cardiologist.'
                },
                { id: 'Thalassemia', label: 'Thalassemia', type: 'dropdown', 
                    options: [
                        { value: '1', label: 'Normal' },
                        { value: '2', label: 'Fixed Defect' },
                        { value: '3', label: 'Reversible Defect' }
                    ],
                    info: 'This refers to a type of stress test called a Thallium scan, which assesses blood flow to the heart muscle.\n• Normal: Blood flow to the heart muscle is normal.\n• Fixed Defect: An area of the heart muscle has reduced blood flow at rest and during exercise.\n• Reversible Defect: An area of the heart has reduced blood flow only during exercise, but normal flow at rest.'
                }
            ]
        },
        cancer: {
            name: 'Breast Cancer',
            endpoint: '/api/predict/cancer',
            batchEndpoint: '/api/predict/cancer/batch',
            positiveClass: 'MALIGNANT',
            negativeClass: 'BENIGN',
            positiveDesc: "Malignant means the tumor is cancerous and can spread to other parts of the body. It requires immediate medical attention and treatment to prevent metastasis.",
            negativeDesc: 'Benign means the tumor is non-cancerous and does not spread to other parts of the body. While it may still require monitoring, it is generally not life-threatening.',
            attributes: [
                { id: 'Clump Thickness', label: 'Clump Thickness', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '0', 
                    info:'Refers to the degree to which cells are clustered together. Higher thickness values may indicate abnormal cell growth or potential malignancy.' 
                },
                { id: 'Uniformity of Cell Size', label: 'Uniformity of Cell Size', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '0', 
                    info:'Measures the consistency in cell sizes within the sample. Significant variation in size may suggest the presence of abnormal or cancerous cells.' 
                },
                { id: 'Uniformity of Cell Shape', label: 'Uniformity of Cell Shape', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '0', 
                    info:'Evaluates the uniformity of cell shapes. Normal cells generally maintain consistent shapes, while irregular shapes may be indicative of malignancy.' 
                },
                { id: 'Marginal Adhesion', label: 'Marginal Adhesion', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '0', 
                    info:'Describes the extent to which cells adhere to one another. Poor adhesion may signify abnormal or invasive cellular behavior.' 
                },
                { id: 'Single Epithelial Cell Size', label: 'Single Epithelial Cell Size', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '0', 
                    info:'Represents the average size of individual epithelial cells. Enlarged epithelial cells are often associated with abnormal cellular activity.' 
                },
                { id: 'Bare Nuclei', label: 'Bare Nuclei', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '0', 
                    info:'Indicates the presence of nuclei without surrounding cytoplasm. A higher count of bare nuclei is commonly observed in malignant samples.' 
                },
                { id: 'Bland Chromatin', label: 'Bland Chromatin', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '0', 
                    info:'Refers to the texture and appearance of the chromatin within the nucleus. Coarse or uneven chromatin patterns may suggest abnormal or cancerous growth.' 
                },
                { id: 'Normal Nucleoli', label: 'Normal Nucleoli', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '0', 
                    info:'Pertains to the visibility and prominence of nucleoli within the nucleus. Prominent or multiple nucleoli are often linked to increased cellular activity, typical of cancerous cells.' 
                },
                { id: 'Mitoses', label: 'Mitoses', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '0', 
                    info:'Measures the frequency of cell division. An elevated mitotic rate reflects rapid cellular proliferation, which may indicate malignant behavior.' 
                }
            ]
        }
    };

    // Disease information content for modals
    const diseaseInfoContent = {
        diabetes: {
            title: 'Diabetes',
            content: `
                <p>This data is available from a general practitioner through a standard medical check-up, blood tests (like an Oral Glucose Tolerance Test), and your patient history.</p>
                <p>This predictor uses your actual, raw test results (ex. glucose level in mg/dL), not a graded scale.</p>
                <p class="citation">Smith, J.W., et al. (1988). Pima Indians Diabetes Database. UCI Machine Learning Repository</p>
            `,
            buttonColor: '#096F29'
        },
        heart: {
            title: 'Heart',
            content: `
                <p>The information needed here must come from a comprehensive cardiac exam by a cardiologist, which includes blood tests, a physical exam, and a cardiac stress test.</p>
                <p>The model uses a mix of direct measurements (like blood pressure) and categories defined by your doctor (like chest pain type). Use the exact values from your medical report.</p>
                <p class="citation">Janosi, A., et al. (1988). Heart Disease Data Set. UCI Machine Learning Repository.</p>
            `,
            buttonColor: '#811111'
        },
        cancer: {
            title: 'Cancer',
            content: `
                <p>These values are highly specialized and can only be found in a pathology report after a fine-needle aspiration (FNA) biopsy. You must get this report from your oncologist or pathologist.</p>
                <p>The 1 to 10 scale for these features was originally developed by Dr. William H. Wolberg, a physician and one of the researchers. He introduced the system by personally assigning each feature an integer value ranging from 1 to 10.</p>
                <p>According to his system, a value of 1 represented a state closest to benign (non-cancerous), while a value of 10 represented the most anaplastic (a severe form of malignant) state.</p>
                <p class="citation">Wolberg, W.H., & Mangasarian, O.L. (1990). Wisconsin Breast Cancer Database. UCI Machine Learning Repository.</p>
            `,
            buttonColor: '#1E2F4E'
        }
    };

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

        // Update chart colors and legend AFTER singlePrediction is initialized
        if (window.singlePrediction) {
            window.singlePrediction.updateChartColors(disease);
        }
    }

    // Back button functionality
    window.goBack = function() {
        // Clear sessionStorage when intentionally leaving the page
        sessionStorage.removeItem('selectedDisease');
        sessionStorage.removeItem('selectedMode');
        window.location.href = 'index.html';
    };

    function handleBeforeUnload(e) {
        e.preventDefault();
        e.returnValue = ''; // Chrome requires returnValue to be set
        return ''; // Some browsers show this message
        
    }

    
    


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

        // Handle form submission
        form.addEventListener('submit', async function(e) {
            e.preventDefault();

            const config = diseaseConfigs[currentDisease];

            // Special validation for cancer sliders
            if (currentDisease === 'cancer') {
                const unmodifiedFields = [];
                config.attributes.forEach(attr => {
                    if (attr.type === 'slider') {
                        const slider = document.getElementById(attr.id);
                        if (slider && slider.value === '0') {
                            unmodifiedFields.push(attr.id);
                        }
                    }
                });

                if (unmodifiedFields.length > 0) {
                    showValidationModal(unmodifiedFields);
                    return;
                }
            }

            // Get form data dynamically - FIXED VERSION
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
                } else if (attr.type === 'slider') {
                    // For sliders, get the value
                    const sliderInput = document.getElementById(attr.id);
                    if (sliderInput) {
                        formData[attr.id] = sliderInput.value;
                    }
                } else {
                    // For regular inputs (text, number)
                    const input = document.getElementById(attr.id);
                    if (input) {
                        formData[attr.id] = input.value;
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
                config.attributes.forEach(attr => {
                    if (attr.type === 'slider') {
                        // Reset sliders to default (0)
                        const input = document.getElementById(attr.id);
                        const valueDisplay = input?.nextElementSibling;
                        if (input && valueDisplay) {
                            input.value = attr.default || '0';
                            valueDisplay.textContent = input.value;
                            if (input.value === '0' && currentDisease === 'cancer') {
                                valueDisplay.classList.add('slider-unmodified');
                            }
                        }
                    } else if (attr.type === 'dropdown') {
                        // Reset dropdowns to default (first disabled option)
                        const select = document.getElementById(attr.id);
                        if (select) {
                            select.selectedIndex = 0; // Select the "Select an option" default
                        }
                    } else if (attr.type === 'radio') {
                        // Uncheck all radio buttons
                        const radioInputs = document.querySelectorAll(`input[name="${attr.id}"]`);
                        radioInputs.forEach(radio => {
                            radio.checked = false;
                        });
                    } else if (attr.type === 'number') {
                        // Reset number inputs to empty
                        const input = document.getElementById(attr.id);
                        if (input) {
                            input.value = '';
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
            if (attr.type === 'radio') {
                const radioGroup = document.createElement('div');
                radioGroup.className = 'radio-group';
                
                attr.options.forEach((option, index) => {
                    const radioOption = document.createElement('div');
                    radioOption.className = 'radio-option';
                    
                    const input = document.createElement('input');
                    input.type = 'radio';
                    input.name = attr.id;
                    input.id = `${attr.id}_${index}`;  // Unique id for each radio button
                    input.value = option.value;
                    input.required = true;
                    
                    const radioLabel = document.createElement('label');
                    radioLabel.htmlFor = `${attr.id}_${index}`;  // Match the radio button's unique id
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
            } else if (attr.type === 'slider') {
                const slider = document.createElement('input');
                slider.type = 'range';
                slider.id = attr.id;
                slider.className = 'slider-input';
                slider.min = attr.min;
                slider.max = attr.max;
                slider.value = attr.default || '0';
                slider.step = '1';
                
                const valueDisplay = document.createElement('span');
                valueDisplay.className = 'slider-value';
                valueDisplay.textContent = slider.value;
                
                // Add unmodified class if value is 0
                if (slider.value === '0' && currentDisease === 'cancer') {
                    valueDisplay.classList.add('slider-unmodified');
                }
                
                slider.addEventListener('input', function() {
                    valueDisplay.textContent = this.value;
                    // Remove unmodified class when user changes the value
                    if (this.value !== '0' && currentDisease === 'cancer') {
                        valueDisplay.classList.remove('slider-unmodified');
                    } 
                });
                
                wrapper.appendChild(slider);
                wrapper.appendChild(valueDisplay);
            } else {
                // Regular number input
                const input = document.createElement('input');
                input.type = attr.type;
                input.id = attr.id;
                input.placeholder = attr.placeholder || '';
                input.min = attr.min;
                input.max = attr.max;
                input.step = attr.step || '1';
                input.required = true;
                
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
            confirmButtonColor: '#1E2F4E',
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
            link.href = `csv-files/diabetes_sample.csv`;
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
        
        // Clear table body
        tableBody.innerHTML = `<tr><td colspan="${config.attributes.length + 1}" class="empty-table-message">No data uploaded yet. Upload a CSV file to see data here.</td></tr>`;
    }

    function handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        console.log('File selected:', file.name, 'Type:', file.type, 'Size:', file.size);

        const reader = new FileReader();
        reader.onload = async function(e) {
            try {
                const csvData = e.target.result;
                const parsedData = parseCSV(csvData);
                
                // Store original count before validation
                const originalCount = parsedData.data.length;
                
                const isValid = await validateData(parsedData);
                
                if (!isValid) {
                    resetFileInput();
                    return;
                }
                
                // Calculate how many records were removed (if any)
                const finalCount = parsedData.data.length;
                const removedCount = originalCount - finalCount;
                
                // Validation succeeded - now we can safely update state
                diseaseStates[currentDisease].batchUploadedData = parsedData;
                diseaseStates[currentDisease].batchPredictedData = null;
                
                displayData(parsedData);
                document.getElementById('predictBtn').style.display = 'block';
                document.getElementById('downloadBtn').style.display = 'none';
                
                // Hide prediction columns
                document.querySelectorAll('.prediction-column').forEach(col => {
                    col.style.display = 'none';
                });
                
                // Show success modal with appropriate message
                if (removedCount > 0) {
                    // Data was cleaned
                    await Swal.fire({
                        icon: 'success',
                        title: 'File Uploaded Successfully!',
                        html: `
                            <div style="text-align: center;">
                                <p>Loaded ${finalCount} valid records.</p>
                                <p style="color: #666; font-size: 14px;">
                                    ${removedCount} invalid record${removedCount > 1 ? 's were' : ' was'} removed.
                                </p>
                            </div>
                        `,
                        confirmButtonColor: getThemeColor()
                    });
                } else {
                    // No cleaning needed
                    await Swal.fire({
                        icon: 'success',
                        title: 'File Uploaded Successfully!',
                        text: `Loaded ${finalCount} records.`,
                        confirmButtonColor: getThemeColor()
                    });
                }

            } catch (error) {
                console.error('CSV parsing error:', error);
                Swal.fire({
                    icon: 'error',
                    title: 'Invalid CSV File',
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
            
            // DON'T reset file input here - let handleFileUpload do it
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
            
            // DON'T reset file input here - let handleFileUpload do it
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
            
            // await Swal.fire({
            //     icon: 'success',
            //     title: 'Data Cleaned!',
            //     text: `Removed ${invalidCount} invalid records. ${validCount} valid records remaining.`,
            //     confirmButtonColor: getThemeColor()
            // });
            
            return true; // Validation passed after cleaning
        } else {
            // User cancelled - DON'T reset file input here, let handleFileUpload do it
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

            // FIXED: Backend returns 'probabilities' not 'percentages'
            predictedData = {
                predictions: result.predictions,
                probabilities: result.probabilities  // This is the correct key from backend
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