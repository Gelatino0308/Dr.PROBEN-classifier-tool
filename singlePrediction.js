document.addEventListener('DOMContentLoaded', () => {
    
    const form = document.getElementById('predictionForm');
    const displayResult = document.getElementById('result');
    let percentText = document.getElementById('percentText');
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
                { id: 'pregnancies', label: 'Number of Pregnancies', placeholder: '0', min: '0', type: 'number' },
                { id: 'plasma', label: 'Plasma Glucose Concentration', placeholder: '0 (mg/dL)', min: '0', type: 'number' },
                { id: 'BP', label: 'Diastolic Blood Pressure', placeholder: '0 (mm Hg)', min: '0', type: 'number' },
                { id: 'skin', label: 'Triceps Skin Fold Thickness', placeholder: '0 (mm)', min: '0', type: 'number' },
                { id: 'insulin', label: '2-Hour Serum Insulin', placeholder: '0 (µU/mL)', min: '0', type: 'number' },
                { id: 'BMI', label: 'Body Mass Index', placeholder: '0.0 (kg/m²)', min: '0', type: 'number', step: 'any' },
                { id: 'pedigree', label: 'Diabetes Pedigree Function', placeholder: '0.000', min: '0', type: 'number', step: 'any' },
                { id: 'age', label: 'Age', placeholder: '0', min: '0', type: 'number' }
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
                { id: 'age', label: 'Age', placeholder: '0', min: '0', type: 'number' },
                { 
                    id: 'sex', 
                    label: 'Sex', 
                    type: 'radio', 
                    options: [
                        { value: '1', label: 'Male' },
                        { value: '0', label: 'Female' }
                    ]
                },
                { 
                    id: 'cp', 
                    label: 'Chest Pain Type', 
                    type: 'dropdown', 
                    options: [
                        { value: '0', label: 'Asymptomatic' },
                        { value: '1', label: 'Atypical Angina' },
                        { value: '2', label: 'Non-anginal Pain' },
                        { value: '3', label: 'Typical Angina' }
                    ]
                },
                { id: 'trestbps', label: 'Resting Blood Pressure', placeholder: '0 (mm Hg)', min: '0', type: 'number' },
                { id: 'chol', label: 'Serum Cholesterol', placeholder: '0 (mg/dL)', min: '0', type: 'number' },
                { 
                    id: 'fbs', 
                    label: 'FBS > 120mg/dL', 
                    type: 'radio', 
                    options: [
                        { value: '1', label: 'True' },
                        { value: '0', label: 'False' }
                    ]
                },
                { 
                    id: 'restecg', 
                    label: 'Resting ECG Results', 
                    type: 'dropdown', 
                    options: [
                        { value: '0', label: 'Normal' },
                        { value: '1', label: 'ST-T Wave Abnormality' },
                        { value: '2', label: 'Left Ventricular Hypertrophy' }
                    ]
                },
                { id: 'thalach', label: 'Maximum Heart Rate', placeholder: '0', min: '0', type: 'number' },
                { 
                    id: 'exang', 
                    label: 'Exercise Induced Angina', 
                    type: 'radio', 
                    options: [
                        { value: '1', label: 'Yes' },
                        { value: '0', label: 'No' }
                    ]
                },
                { id: 'oldpeak', label: 'ST Depression (Oldpeak)', placeholder: '0.0', min: '0', type: 'number', step: 'any' },
                { 
                    id: 'slope', 
                    label: 'Slope of Peak Exercise ST', 
                    type: 'dropdown', 
                    options: [
                        { value: '0', label: 'Downsloping' },
                        { value: '1', label: 'Flat' },
                        { value: '2', label: 'Upsloping' }
                    ]
                },
                { id: 'ca', label: 'Number of Major Vessels', placeholder: '0-4', min: '0', max: '4', type: 'slider', default: '0' },
                { 
                    id: 'thal', 
                    label: 'Thalassemia', 
                    type: 'dropdown', 
                    options: [
                        { value: '1', label: 'Normal' },
                        { value: '2', label: 'Fixed Defect' },
                        { value: '3', label: 'Reversible Defect' }
                    ]
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
                { id: 'clump_thickness', label: 'Clump Thickness', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '5' },
                { id: 'uniformity_cell_size', label: 'Uniformity of Cell Size', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '5' },
                { id: 'uniformity_cell_shape', label: 'Uniformity of Cell Shape', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '5' },
                { id: 'marginal_adhesion', label: 'Marginal Adhesion', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '5' },
                { id: 'single_epithelial_cell_size', label: 'Single Epithelial Cell Size', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '5' },
                { id: 'bare_nuclei', label: 'Bare Nuclei', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '5' },
                { id: 'bland_chromatin', label: 'Bland Chromatin', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '5' },
                { id: 'normal_nucleoli', label: 'Normal Nucleoli', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '5' },
                { id: 'mitoses', label: 'Mitoses', placeholder: '1-10', min: '1', max: '10', type: 'slider', default: '5' }
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
            document.querySelector('.result-section h1').textContent = state.resultData.prediction === 1 ? config.positiveClass : config.negativeClass;
            document.querySelector('.result-section p').textContent = state.resultData.prediction === 1 ? config.positiveDesc : config.negativeDesc;
            percentText.textContent = state.resultData.percentage + '%';
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
            label.setAttribute('for', attr.id);
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
                if (attr.max) input.max = attr.max;
                if (attr.step) input.step = attr.step;
                input.required = true;
                inputWrapper.appendChild(input);
            }
            
            inputsContainer.appendChild(inputWrapper);
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
            description.textContent = result.prediction === 1 ? config.positiveDesc : config.negativeDesc;

            // Update chart with the prediction percentage
            percentage = result.percentage;
            percentText.textContent = percentage + '%';
            
            // Make result details appear after first submit
            displayResult.style.display = 'flex';
            
            // Update chart data
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