document.addEventListener('DOMContentLoaded', () => {
    
    const form = document.getElementById('predictionForm');
    const displayResult = document.getElementById('result');
    let percentText = document.getElementById('percentText');
    let chartLabel =  document.getElementById('chartLabel');
    let percentage = 0;
    let currentDisease = 'diabetes'; // Default disease

    // State management object to hold data for each tab
    const diseaseStates = {
        diabetes: { formData: {}, resultData: null },
        heart: { formData: {}, resultData: null },
        cancer: { formData: {}, resultData: null }
    };

    // Disease configurations
    const diseaseConfigs = {
        diabetes: {
            name: 'Diabetes',
            endpoint: '/api/predict/diabetes',
            positiveClass: 'DIABETIC',
            negativeClass: 'NON-DIABETIC',
            positiveDesc: "Diabetic means the person has diabetes, a chronic disease that affects how your body turns food into energy. It occurs when your pancreas doesn't make enough insulin or your cells don't respond to insulin properly.",
            negativeDesc: 'Non-diabetic means the absence of diabetes. Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood glucose.',
            attributes: [
                { id: 'pregnancies', label: 'Number of Pregnancies', placeholder: '0', min: '0', type: 'number', 
                    info: 'If you have been pregnant twice, you would enter "2." If you have never been pregnant, you would enter "0."' 
                },
                { id: 'plasma', label: 'Plasma Glucose Concentration', placeholder: '0 (mg/dL)', min: '0', type: 'number', 
                    info: 'This measures the amount of sugar in your blood. You will need to get this value from a recent blood test, often called a blood sugar test or glucose test. Look for a result listed as "Fasting Plasma Glucose" or similar, which is measured in milligrams per deciliter (mg/dL).'
                },
                { id: 'BP', label: 'Diastolic Blood Pressure', placeholder: '0 (mm Hg)', min: '0', type: 'number',
                    info: 'This is the second, or lower, number in a blood pressure reading. A reading is typically written as two numbers, like "120/80." In this example, "80" is the diastolic pressure. You can get this from a recent doctor\'s visit or a home blood pressure monitor.'
                },
                { id: 'skin', label: 'Triceps Skin Fold Thickness', placeholder: '0 (mm)', min: '0', type: 'number',
                    info: 'This value is a way to estimate the amount of body fat by measuring the thickness of a fold of skin and fat on the back of your upper arm. This measurement is usually taken with a special tool called a caliper. You will need to get this value from your doctor.'    
                },
                { id: 'insulin', label: '2-Hour Serum Insulin', placeholder: '0 (µU/mL)', min: '0', type: 'number',
                    info: 'This measures the amount of insulin in your blood specifically two hours after you\'ve taken a glucose tolerance test. It shows how well your body processes sugar over time. This value should be obtained from a specific blood test.'
                },
                { id: 'BMI', label: 'Body Mass Index', placeholder: '0.0 (kg/m²)', min: '0', type: 'number', step: 'any',
                    info: 'Your BMI is a value calculated from your weight and height that helps determine if you are at a healthy weight. To find your BMI, you can use an online calculator. Simply enter your height and weight, and the calculator will provide your BMI value. For example, if you weigh 150 lbs and are 5\'5" tall, your BMI is approximately 25.'
                },
                { id: 'pedigree', label: 'Diabetes Pedigree Function', placeholder: '0.000', min: '0', type: 'number', step: 'any',
                    info: 'This is a complex score that quantifies the genetic risk of diabetes based on your family history. You won\'t have a number for this yourself. This value is typically calculated by the diagnostic tool based on the family history information you provide, such as whether your parents or siblings have diabetes.'
                },
                { id: 'age', label: 'Age', placeholder: '0', min: '0', type: 'number',
                    info:'This is your current age.'
                }
            ]
        },
        heart: {
            name: 'Heart Disease',
            endpoint: '/api/predict/heart',
            positiveClass: 'POSITIVE',
            negativeClass: 'NEGATIVE',
            positiveDesc: "Positive means the presence of heart disease. Heart disease refers to several types of heart conditions that affect the heart's ability to function normally. It includes coronary artery disease, heart rhythm problems, and heart defects.",
            negativeDesc: 'Negative means the absence of cardiovascular conditions. A healthy heart efficiently pumps blood throughout the body, delivering oxygen and nutrients to organs and tissues.',
            attributes: [
                { id: 'age', label: 'Age', placeholder: '0', min: '0', type: 'number',
                    info: 'This is your current age.'
                },
                { id: 'sex', label: 'Sex', type: 'radio', 
                    options: [
                        { value: '1', label: 'Male' },
                        { value: '0', label: 'Female' }
                    ],
                    info: 'This refers to your biological sex.'
                },
                { id: 'cp', label: 'Chest Pain Type', type: 'dropdown', 
                    options: [
                        { value: '0', label: 'Asymptomatic' },
                        { value: '1', label: 'Atypical Angina' },
                        { value: '2', label: 'Non-anginal Pain' },
                        { value: '3', label: 'Typical Angina' }
                    ],
                    info: 'Common types are:\n• Typical Angina: Chest pain caused by reduced blood flow to the heart\n• Atypical Angina: Chest discomfort that doesn\'t follow typical angina patterns\n• Non-anginal Pain: Chest pain not related to heart conditions\n• Asymptomatic: No chest pain symptoms'
                },
                { id: 'trestbps', label: 'Resting Blood Pressure', placeholder: '0 (mm Hg)', min: '0', type: 'number',
                    info: 'This is the top number of your blood pressure reading, measured while you are at rest. It is measured in millimeters of mercury (mm Hg).'
                },
                { id: 'chol', label: 'Serum Cholesterol', placeholder: '0 (mg/dL)', min: '0', type: 'number',
                    info: 'This is the amount of total cholesterol in your blood. It is measured in milligrams per deciliter (mg/dL).'
                },
                { id: 'fbs', label: 'FBS > 120mg/dL', type: 'radio', 
                    options: [
                        { value: '1', label: 'True' },
                        { value: '0', label: 'False' }
                    ],
                    info: 'This indicates whether your fasting blood sugar is greater than 120 mg/dL. This is a common threshold for diagnosing prediabetes or diabetes.\n• True: Your fasting blood sugar is greater than 120 mg/dL.\n• False: Your fasting blood sugar is 120 mg/dL or less.'
                },
                { id: 'restecg', label: 'Resting ECG Results', type: 'dropdown', 
                    options: [
                        { value: '0', label: 'Normal' },
                        { value: '1', label: 'ST-T Wave Abnormality' },
                        { value: '2', label: 'Left Ventricular Hypertrophy' }
                    ],
                    info: 'This is a record of your heart\'s electrical activity while you are at rest. You will need a recent ECG report.\n• Normal: No significant abnormalities.\n• ST-T Wave Abnormality: Minor changes that could indicate a heart issue.\n• Left Ventricular Hypertrophy (LVH): Thickening of the heart\'s main pumping chamber.'
                },
                { id: 'thalach', label: 'Maximum Heart Rate', placeholder: '0', min: '0', type: 'number',
                    info:'This is the highest heart rate you reached during a stress or exercise test. This measurement is often taken on a treadmill or stationary bike while your heart rate is monitored.'
                },
                { id: 'exang', label: 'Exercise Induced Angina', type: 'radio', 
                    options: [
                        { value: '1', label: 'Yes' },
                        { value: '0', label: 'No' }
                    ],
                    info: 'This indicates whether you experienced chest pain during physical exercise.\n• True: You experienced chest pain during exercise.\n• False: You did not experience chest pain during exercise.'
                },
                { id: 'oldpeak', label: 'ST Depression (Oldpeak)', placeholder: '0.0', min: '0', type: 'number', step: 'any',
                    info: 'This measures the amount of depression in the ST segment of your ECG during exercise, which can be a sign of reduced blood flow to the heart. The value is measured in millimeters.'
                },
                { id: 'slope', label: 'Slope of Peak Exercise ST', type: 'dropdown', 
                    options: [
                        { value: '0', label: 'Downsloping' },
                        { value: '1', label: 'Flat' },
                        { value: '2', label: 'Upsloping' }
                    ],
                    info:'Describes the slope of ST segment on your ECG during an exercise stress test.\n• Upsloping: The ST segment goes up.\n• Flat: The ST segment is horizontal.\n• Downsloping: The ST segment goes down. A downsloping or flat slope can be a sign of heart disease.'
                },
                { id: 'ca', label: 'Number of Major Vessels', placeholder: '0-4', min: '0', max: '4', type: 'slider', default: '0',
                    info: 'This refers to the number of major blood vessels (0 to 3) that are a significant percentage narrowed as seen in a coronary angiography. This value is provided by a cardiologist.'
                },
                { id: 'thal', label: 'Thalassemia', type: 'dropdown', 
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
            positiveClass: 'MALIGNANT',
            negativeClass: 'BENIGN',
            positiveDesc: "Malignant means the tumor is cancerous and can spread to other parts of other parts of the body. It requires immediate medical attention and treatment to prevent metastasis.",
            negativeDesc: 'Benign means the tumor is non-cancerous and does not spread to other parts of the body. While it may still require monitoring, it is generally not life-threatening.',
            attributes: [
                { id: 'clump_thickness', label: 'Clump Thickness', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '5', info:'' },
                { id: 'uniformity_cell_size', label: 'Uniformity of Cell Size', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '5', info:'' },
                { id: 'uniformity_cell_shape', label: 'Uniformity of Cell Shape', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '5', info:'' },
                { id: 'marginal_adhesion', label: 'Marginal Adhesion', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '5', info:'' },
                { id: 'single_epithelial_cell_size', label: 'Single Epithelial Cell Size', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '5', info:'' },
                { id: 'bare_nuclei', label: 'Bare Nuclei', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '5', info:'' },
                { id: 'bland_chromatin', label: 'Bland Chromatin', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '5', info:'' },
                { id: 'normal_nucleoli', label: 'Normal Nucleoli', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '5', info:'' },
                { id: 'mitoses', label: 'Mitoses', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '5', info:'' }
            ]
        }
    };

    // Initialize chart data
    const data = {
        datasets: [{
            data: [percentage, 100 - percentage],
            backgroundColor: ['#ffffff', '#8c5700'],
            borderWidth: 0,
            cutout: '65%',
        }]
    };

    const shadowPlugin = {
        id: 'shadow',
        beforeDatasetDraw(chart, args) {
            const ctx = chart.ctx;
            const datasetIndex = args.index;

            if (datasetIndex === 0) {
                ctx.save();
                ctx.shadowColor = 'rgba(0, 0, 0, 0.4)';
                ctx.shadowBlur = 10;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 0;
            }
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
                                return `${config.positiveClass} Probability: ${value}%`;
                            } else {
                                return `${config.negativeClass} Probability: ${value}%`;
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

    // Function to update instruction text
    function updateInstruction(diseaseName) {
        const instruction = document.querySelector('.instruction');
        instruction.innerHTML = `<span>Enter the required information</span> related to the diagnostic measurements of ${diseaseName.toLowerCase()} in the form below. Once you're done, click the <span>"Predict Class"</span> button to see the result. This will help you check if the case may be classified as <span>${diseaseConfigs[currentDisease].positiveClass.toLowerCase()}</span> or <span>${diseaseConfigs[currentDisease].negativeClass.toLowerCase()}</span>.`;
    }

    // Function to save the state of the current tab
    function saveCurrentState() {
        if (!currentDisease) return;
        const currentState = diseaseStates[currentDisease];
        
        // Save form data
        const currentForm = document.getElementById('predictionForm');
        const formData = new FormData(currentForm);
        for (let [key, value] of formData.entries()) {
            currentState.formData[key] = value;
        }
    }

    // Function to restore the state of a tab
    function restoreState(disease) {
        const state = diseaseStates[disease];
        
        // Restore form data
        for (const [key, value] of Object.entries(state.formData)) {
            const input = form.elements[key];
            if (input) {
                if (input.type === 'radio') {
                    // Handle radio buttons
                    const radioInput = form.querySelector(`input[name="${key}"][value="${value}"]`);
                    if (radioInput) radioInput.checked = true;
                } else if (input.type === 'range') {
                    // Handle sliders
                    input.value = value;
                    const valueDisplay = document.getElementById(`${key}_value`);
                    if (valueDisplay) valueDisplay.textContent = value;
                } else {
                    // Handle regular inputs and dropdowns
                    input.value = value;
                }
            }
        }

        // Restore prediction result
        if (state.resultData) {
            const config = diseaseConfigs[disease];
            
            if (state.resultData.prediction === 1) {
                document.querySelector('.result-section h1').textContent = config.positiveClass;
                document.querySelector('.result-section p').textContent = config.positiveDesc;
                percentText.textContent = state.resultData.percentage + '%';
            }
            else {
                document.querySelector('.result-section h1').textContent = config.negativeClass;
                document.querySelector('.result-section p').textContent = config.negativeDesc;
                percentText.textContent = (100 - state.resultData.percentage) + '%';
            }

            chart.data.datasets[0].data = [state.resultData.percentage, 100 - state.resultData.percentage];
            chart.update('none');
            displayResult.style.display = 'flex';
        } else {
            // If no result, reset the view
            displayResult.style.display = 'none';
            percentText.textContent = '--%';
            chart.data.datasets[0].data = [0, 100];
            chart.update('none');
        }
    }

    // Function to update page content based on selected disease
    function updatePageContent(disease) {
        // Save state of the previous tab before switching
        saveCurrentState();

        currentDisease = disease;
        const config = diseaseConfigs[disease];
        
        // Update page title only
        document.title = `Dr. PROBEN | ${config.name} Prediction`;
        
        // Update instruction text
        updateInstruction(config.name);
        
        // Update form (rebuilds inputs)
        updateForm(config.attributes);
        
        // Restore the state for the new tab (form values and results)
        restoreState(disease);
        
        // Update chart colors based on disease theme
        updateChartColors(disease);
        
        // Update submit button color based on theme
        updateSubmitButtonColor(disease);
    }

    // Function to update form fields
    function updateForm(attributes) {
        const labelsContainer = document.querySelector('.labels-container');
        const inputsContainer = document.querySelector('.inputs-container');
        
        // Clear existing content
        labelsContainer.innerHTML = '';
        inputsContainer.innerHTML = '';
        
        // Add new fields
        attributes.forEach(attr => {
            // Create label
            const label = document.createElement('label');
            label.textContent = attr.label;
            labelsContainer.appendChild(label);
            
            // Create input container
            const inputWrapper = document.createElement('div');
            inputWrapper.className = 'input-wrapper';
            
            if (attr.type === 'slider') {
                // Create slider input
                const input = document.createElement('input');
                input.type = 'range';
                input.id = attr.id;
                input.name = attr.id;
                input.min = attr.min;
                input.max = attr.max;
                input.value = attr.default || attr.min;
                input.required = true;
                input.className = 'slider-input';

                label.setAttribute('for', attr.id);
                
                // Create value display
                const valueDisplay = document.createElement('span');
                valueDisplay.id = `${attr.id}_value`;
                valueDisplay.className = 'slider-value';
                valueDisplay.textContent = attr.default || attr.min;
                
                // Add event listener to update display
                input.addEventListener('input', function() {
                    valueDisplay.textContent = this.value;
                });
                
                inputWrapper.appendChild(input);
                inputWrapper.appendChild(valueDisplay);
            } else if (attr.type === 'radio') {
                // Create radio button group
                const radioGroup = document.createElement('div');
                radioGroup.className = 'radio-group';
                
                attr.options.forEach(option => {
                    const radioWrapper = document.createElement('div');
                    radioWrapper.className = 'radio-option';
                    
                    const radioInput = document.createElement('input');
                    radioInput.type = 'radio';
                    radioInput.id = `${attr.id}_${option.value}`;
                    radioInput.name = attr.id;
                    radioInput.value = option.value;
                    radioInput.required = true;
                    
                    const radioLabel = document.createElement('label');
                    radioLabel.setAttribute('for', `${attr.id}_${option.value}`);
                    radioLabel.textContent = option.label;
                    
                    radioWrapper.appendChild(radioInput);
                    radioWrapper.appendChild(radioLabel);
                    radioGroup.appendChild(radioWrapper);
                });
                
                inputWrapper.appendChild(radioGroup);
            } else if (attr.type === 'dropdown') {
                // Create dropdown/select
                const select = document.createElement('select');
                select.id = attr.id;
                select.name = attr.id;
                select.required = true;
                select.className = 'dropdown-input';

                label.setAttribute('for', attr.id);
                
                // Add default option
                const defaultOption = document.createElement('option');
                defaultOption.value = '';
                defaultOption.textContent = 'Select an option';
                defaultOption.disabled = true;
                defaultOption.selected = true;
                select.appendChild(defaultOption);
                
                // Add options
                attr.options.forEach(option => {
                    const optionElement = document.createElement('option');
                    optionElement.value = option.value;
                    optionElement.textContent = option.label;
                    select.appendChild(optionElement);
                });
                
                inputWrapper.appendChild(select);
            } else {
                // Create regular input
                const input = document.createElement('input');
                input.type = attr.type;
                input.id = attr.id;
                input.name = attr.id;
                input.placeholder = attr.placeholder;
                input.min = attr.min;

                label.setAttribute('for', attr.id);
                
                if (attr.max) input.max = attr.max;
                if (attr.step) input.step = attr.step;
                input.required = true;
                inputWrapper.appendChild(input);
            }
            
            inputsContainer.appendChild(inputWrapper);
        });

        document.querySelectorAll('.labels-container label').forEach((label, index) => {
            const attr = attributes[index];
            tippy(label, {
                content: attr.info.replace(/\n/g, '<br>'), 
                theme: 'light',
                allowHTML: true 
            });
        });
    }

    // Function to update chart colors based on disease theme
    function updateChartColors(disease) {
        const themeColors = {
            diabetes: ['#ffffff', '#8c5700'],
            heart: ['#ffffff', '#5a0c0c'],
            cancer: ['#ffffff', '#032f5c']
        };
        
        chart.data.datasets[0].backgroundColor = themeColors[disease];
        chart.update('none');
    }

    // Function to update submit button color based on theme
    function updateSubmitButtonColor(disease) {
        const submitButton = document.querySelector('input[type="submit"]');
        const themeColors = {
            diabetes: '#DE9B1E',
            heart: '#811111',
            cancer: '#044786'
        };
        
        submitButton.style.color = themeColors[disease];
    }

    // Handle disease button clicks
    document.querySelectorAll('.problem-btns-container button').forEach(button => {
        button.addEventListener('click', function() {
            // Remove selected class from all buttons
            document.querySelectorAll('.problem-btns-container button').forEach(btn => {
                btn.classList.remove('selected-problem');
            });
            
            // Add selected class to clicked button
            this.classList.add('selected-problem');
            
            // Update body theme
            document.body.className = this.className.split(' ')[0];
            
            // Determine disease type
            if (this.classList.contains('theme-diabetes')) {
                updatePageContent('diabetes');
            } else if (this.classList.contains('theme-heart')) {
                updatePageContent('heart');
            } else if (this.classList.contains('theme-cancer')) {
                updatePageContent('cancer');
            }
        });
    });

    // Handle form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const config = diseaseConfigs[currentDisease];
        
        // Get form data dynamically
        const formData = {};
        config.attributes.forEach(attr => {
            const element = document.getElementById(attr.id);
            if (attr.type === 'radio') {
                const checkedRadio = form.querySelector(`input[name="${attr.id}"]:checked`);
                formData[attr.id] = checkedRadio ? checkedRadio.value : '';
            } else {
                formData[attr.id] = element ? element.value : '';
            }
        });

        try {
            // Show loading state
            percentText.textContent = 'Loading...';
            percentText.style.fontSize = '24px';
            chartLabel.style.display = 'none';
            
            // Send data to API
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

            // Save the result to the state object
            diseaseStates[currentDisease].resultData = result;
            
            // Update the prediction text
            document.querySelector('.result-section h1').textContent = 
                result.prediction === 1 ? config.positiveClass : config.negativeClass;
            
            // Update the description based on the prediction
            const description = document.querySelector('.result-section p');

            percentage = result.percentage;
            
            // Update chart with the prediction percentage
            if (result.prediction === 1) {
                description.textContent = config.positiveDesc;
                percentText.textContent = percentage + '%';
            }
            else {
                description.textContent = config.negativeDesc;
                percentText.textContent = (100 - percentage) + '%';
            }

            percentText.style.fontSize = '32px';
            chartLabel.style.display = 'block';
            
            
            // Make result details appear after first submit
            displayResult.style.display = 'flex';
            
            // Update chart data
            percentage = result.percentage;
            chart.data.datasets[0].data = [percentage, 100 - percentage];
            chart.update('active');
            
        } catch (error) {
            console.error('Error:', error);
            percentText.textContent = 'Error!';
        }
    });

    // Handle hiding of result details after reset
    form.addEventListener('reset', () => {
        // Clear the state for the current disease on reset
        diseaseStates[currentDisease].formData = {};
        diseaseStates[currentDisease].resultData = null;
        
        // Reset form elements to defaults for all diseases
        const config = diseaseConfigs[currentDisease];
        
        // Use setTimeout to ensure form reset happens first
        setTimeout(() => {
            config.attributes.forEach(attr => {
                if (attr.type === 'radio') {
                    // Uncheck all radio buttons
                    const radioInputs = form.querySelectorAll(`input[name="${attr.id}"]`);
                    radioInputs.forEach(radio => radio.checked = false);
                } else if (attr.type === 'dropdown') {
                    // Reset dropdown to default state
                    const select = document.getElementById(attr.id);
                    if (select) {
                        select.selectedIndex = 0; // Select the "Select an option" placeholder
                    }
                } else if (attr.type === 'slider') {
                    // Reset slider to default value
                    const input = document.getElementById(attr.id);
                    const valueDisplay = document.getElementById(`${attr.id}_value`);
                    if (input && valueDisplay) {
                        const defaultValue = attr.default || attr.min;
                        input.value = defaultValue;
                        valueDisplay.textContent = defaultValue;
                    }
                } else if (attr.type === 'number') {
                    // Reset number inputs to empty
                    const input = document.getElementById(attr.id);
                    if (input) {
                        input.value = '';
                    }
                }
            });
        }, 0);
        
        // Reset result display
        displayResult.style.display = 'none';
        percentText.textContent = '--%';
        chart.data.datasets[0].data = [0, 100];
        chart.update('none');
    });
    
    const storedDisease = sessionStorage.getItem('selectedDisease');

    if (storedDisease && diseaseConfigs[storedDisease]) {
        sessionStorage.removeItem('selectedDisease');

        // Reset button selection
        document.querySelectorAll('.problem-btns-container button').forEach(btn => {
            btn.classList.remove('selected-problem');
        });
        
        const targetButton = document.querySelector(`.theme-${storedDisease}`);
        if (targetButton) {
            targetButton.classList.add('selected-problem');
            document.body.className = `theme-${storedDisease}`;
            updatePageContent(storedDisease);
        }
    } else {
        // Default to diabetes
        updatePageContent('diabetes');
    }

    // Function to set active navigation link
    function setActiveNavLink() {
        const currentPage = window.location.pathname.split('/').pop();
        const navLinks = document.querySelectorAll('.links-container a');
        
        // Remove active class from all links
        navLinks.forEach(link => link.classList.remove('active'));
        
        // Set active based on current page
        if (currentPage === 'singlePrediction.html' || currentPage === '' || currentPage === 'index.html') {
            document.getElementById('single-prediction-link')?.classList.add('active');
        } else if (currentPage === 'batchPrediction.html') {
            document.getElementById('batch-prediction-link')?.classList.add('active');
        } else if (currentPage === 'about.html') {
            document.getElementById('about-link')?.classList.add('active');
        }
    }
    
    // Call the function to set active link
    setActiveNavLink();
});