document.addEventListener('DOMContentLoaded', () => {
    let currentDisease = 'diabetes';
    let uploadedData = null;
    let predictedData = null;

    // State management object to hold data for each disease tab
    const diseaseStates = {
        diabetes: { uploadedData: null, predictedData: null },
        heart: { uploadedData: null, predictedData: null },
        cancer: { uploadedData: null, predictedData: null }
    };

    // Disease configurations matching single prediction
    const diseaseConfigs = {
        diabetes: {
            name: 'Diabetes',
            endpoint: '/api/predict/diabetes/batch',
            positiveClass: 'DIABETIC',
            negativeClass: 'NON-DIABETIC',
            attributes: [
                { id: 'pregnancies', label: 'Pregnancies', type: 'number', min: 0, max: 20 },
                { id: 'plasma', label: 'Plasma Glucose Concentration', type: 'number', min: 0, max: 300 },
                { id: 'BP', label: 'Diastolic Blood Pressure', type: 'number', min: 0, max: 200 },
                { id: 'skin', label: 'Triceps Skin Fold Thickness', type: 'number', min: 0, max: 100 },
                { id: 'insulin', label: '2-Hour Serum Insulin', type: 'number', min: 0, max: 1000 },
                { id: 'BMI', label: 'Body Mass Index', type: 'number', min: 0, max: 70, step: 0.1 },
                { id: 'pedigree', label: 'Diabetes Pedigree Function', type: 'number', min: 0, max: 5, step: 0.001 },
                { id: 'age', label: 'Age', type: 'number', min: 0, max: 120 }
            ]
        },
        heart: {
            name: 'Heart Disease',
            endpoint: '/api/predict/heart/batch',
            positiveClass: 'HEART DISEASE',
            negativeClass: 'NO HEART DISEASE',
            attributes: [
                { id: 'age', label: 'Age', type: 'number', min: 0, max: 120 },
                { id: 'sex', label: 'Sex', type: 'categorical', values: [0, 1], labels: ['Female', 'Male'] },
                { id: 'cp', label: 'Chest Pain Type', type: 'categorical', values: [0, 1, 2, 3], labels: ['Asymptomatic', 'Atypical Angina', 'Non-anginal Pain', 'Typical Angina'] },
                { id: 'trestbps', label: 'Resting Blood Pressure', type: 'number', min: 0, max: 300 },
                { id: 'chol', label: 'Serum Cholesterol', type: 'number', min: 0, max: 1000 },
                { id: 'fbs', label: 'FBS > 120mg/dL', type: 'categorical', values: [0, 1], labels: ['False', 'True'] },
                { id: 'restecg', label: 'Resting ECG Results', type: 'categorical', values: [0, 1, 2], labels: ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'] },
                { id: 'thalach', label: 'Maximum Heart Rate', type: 'number', min: 0, max: 250 },
                { id: 'exang', label: 'Exercise Induced Angina', type: 'categorical', values: [0, 1], labels: ['No', 'Yes'] },
                { id: 'oldpeak', label: 'ST Depression (Oldpeak)', type: 'number', min: 0, max: 10, step: 0.1 },
                { id: 'slope', label: 'Slope of Peak Exercise ST', type: 'categorical', values: [0, 1, 2], labels: ['Downsloping', 'Flat', 'Upsloping'] },
                { id: 'ca', label: 'Number of Major Vessels', type: 'categorical', values: [0, 1, 2, 3, 4], labels: ['0', '1', '2', '3', '4'] },
                { id: 'thal', label: 'Thalassemia', type: 'categorical', values: [1, 2, 3], labels: ['Normal', 'Fixed Defect', 'Reversible Defect'] }
            ]
        },
        cancer: {
            name: 'Breast Cancer',
            endpoint: '/api/predict/cancer/batch',
            positiveClass: 'MALIGNANT',
            negativeClass: 'BENIGN',
            attributes: [
                { id: 'clump_thickness', label: 'Clump Thickness', type: 'categorical', values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] },
                { id: 'uniformity_cell_size', label: 'Uniformity of Cell Size', type: 'categorical', values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] },
                { id: 'uniformity_cell_shape', label: 'Uniformity of Cell Shape', type: 'categorical', values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] },
                { id: 'marginal_adhesion', label: 'Marginal Adhesion', type: 'categorical', values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] },
                { id: 'single_epithelial_cell_size', label: 'Single Epithelial Cell Size', type: 'categorical', values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] },
                { id: 'bare_nuclei', label: 'Bare Nuclei', type: 'categorical', values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] },
                { id: 'bland_chromatin', label: 'Bland Chromatin', type: 'categorical', values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] },
                { id: 'normal_nucleoli', label: 'Normal Nucleoli', type: 'categorical', values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] },
                { id: 'mitoses', label: 'Mitoses', type: 'categorical', values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] }
            ]
        }
    };

    // DOM elements
    const uploadBtn = document.getElementById('uploadBtn');
    const checkValuesBtn = document.getElementById('checkValuesBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const predictBtn = document.getElementById('predictBtn');
    const fileInput = document.getElementById('fileInput');
    const tableHeader = document.getElementById('tableHeader');
    const tableBody = document.getElementById('tableBody');

    // Initialize page
    function initializePage() {
        // Set initial theme class on body
        // document.body.className = 'theme-diabetes'; // Apply initial theme

        // Check if there's a stored disease selection from navigation
        const storedDisease = sessionStorage.getItem('selectedDisease');
        if (storedDisease && diseaseConfigs[storedDisease]) {
            currentDisease = storedDisease;
            
            // Update button selection
            document.querySelectorAll('.problem-btns-container button').forEach(btn => {
                btn.classList.remove('selected-problem');
            });
            
            const targetButton = document.querySelector(`.theme-${storedDisease}`);
            if (targetButton) {
                targetButton.classList.add('selected-problem');
                document.body.className = `theme-${storedDisease}`;
            }
            
            // Clear the stored disease
            sessionStorage.removeItem('selectedDisease');
        }

        updatePageContent(currentDisease);
        setupEventListeners();
    }

    // Save current state before switching
    function saveCurrentState() {
        if (!currentDisease) return;
        
        const currentState = diseaseStates[currentDisease];
        currentState.uploadedData = uploadedData;
        currentState.predictedData = predictedData;
    }

    // Restore state for selected disease
    function restoreState(disease) {
        const state = diseaseStates[disease];
        
        // Restore data
        uploadedData = state.uploadedData;
        predictedData = state.predictedData;
        
        // Restore UI based on data state
        if (uploadedData) {
            displayData(uploadedData, !!predictedData);
            
            if (predictedData) {
                // Show download button, hide predict button
                predictBtn.style.display = 'none';
                downloadBtn.style.display = 'block';
            } else {
                // Show predict button, hide download button
                predictBtn.style.display = 'block';
                downloadBtn.style.display = 'none';
            }
        } else {
            // No data, reset UI
            resetUI();
        }
    }

    // Update page content based on selected disease
function updatePageContent(disease) {

    // Save current state before switching
    saveCurrentState();

    currentDisease = disease;
    const config = diseaseConfigs[disease];
    
    // Update page title
    document.title = `Dr. PROBEN | ${config.name} Batch Prediction`;
    
    // Update instruction text
    updateInstruction(config.name);
    
    // Update table headers
    updateTableHeaders(config.attributes);

    // Restore state for the new disease
    restoreState(disease);

    // Apply theme class to body
    document.body.className = `theme-${disease}`;
    
    // // Reset data
    // uploadedData = null;
    // predictedData = null;
    
    // // Reset UI
    // resetUI();
    
    // // Apply theme class to body if not already applied
    // if (!document.body.className.includes('theme-')) {
    //     const themeClass = `theme-${disease}`;
    //     document.body.className = themeClass;
    // }
}

    // Update instruction text
    function updateInstruction(diseaseName) {
        const instruction = document.querySelector('.instruction');
        instruction.innerHTML = `<span>Upload</span> (via <span>"Upload Data"</span>) a CSV format file with the same headers as the table below and the has valid values (click <span>"Check Valid Attribute Values"</span> button for more details). The uploaded data will fill the table below, and clicking the <span>"Predict Class"</span> button will fill the classification result for each row. You can also download the CSV file with the classification result column added.`;
    }

    // Update table headers
    function updateTableHeaders(attributes) {
        tableHeader.innerHTML = '';
        
        // Add prediction column header (initially hidden)
        const predictionHeader = document.createElement('th');
        predictionHeader.textContent = 'Prediction';
        predictionHeader.className = 'prediction-column';
        predictionHeader.style.display = 'none';
        tableHeader.appendChild(predictionHeader);
        
        // Add attribute headers
        attributes.forEach(attr => {
            const th = document.createElement('th');
            th.textContent = attr.label;
            tableHeader.appendChild(th);
        });
        
        // Clear table body
        tableBody.innerHTML = '<tr><td colspan="' + (attributes.length + 1) + '" class="empty-table-message">No data uploaded yet. Upload a CSV file to see data here.</td></tr>';
    }

    // Reset UI state
    function resetUI() {
        predictBtn.style.display = 'none';
        downloadBtn.style.display = 'none';
        document.querySelector('.prediction-column').style.display = 'none';
    }

    // Setup event listeners
    function setupEventListeners() {
        // Disease button clicks
        document.querySelectorAll('.problem-btns-container button').forEach(button => {
            button.addEventListener('click', function() {
                document.querySelectorAll('.problem-btns-container button').forEach(btn => {
                    btn.classList.remove('selected-problem');
                });
                
                this.classList.add('selected-problem');
                document.body.className = this.className.split(' ')[0];
                
                if (this.classList.contains('theme-diabetes')) {
                    updatePageContent('diabetes');
                } else if (this.classList.contains('theme-heart')) {
                    updatePageContent('heart');
                } else if (this.classList.contains('theme-cancer')) {
                    updatePageContent('cancer');
                }
            });
        });

        // Upload button
    uploadBtn.addEventListener('click', () => {
        resetFileInput(); // Reset file input before opening dialog
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

    // Update the main file upload handler to handle the new async validation
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    console.log('File selected:', file.name, 'Type:', file.type, 'Size:', file.size);

    // Reset current data (but preserve other diseases' states)
    uploadedData = null;
    predictedData = null;
    diseaseStates[currentDisease].uploadedData = null;
    diseaseStates[currentDisease].predictedData = null;
    
    // Reset UI state
    predictBtn.style.display = 'none';
    downloadBtn.style.display = 'none';
    document.querySelector('.prediction-column').style.display = 'none';

    const reader = new FileReader();
    reader.onload = async function(e) {
        try {
            const csvData = e.target.result;
            console.log('Raw CSV data (first 200 chars):', csvData.substring(0, 200));
            
            const parsedData = parseCSV(csvData);
            console.log('Parsed data:', parsedData);
            
            // Validate data (now async)
            const isValid = await validateData(parsedData);
            
            if (isValid) {
                // Check if uploadedData was set during validation (cleaning process)
                if (!uploadedData) {
                    // If validation passed without needing to clean data
                    uploadedData = parsedData;

                    // Save to state immediately
                    diseaseStates[currentDisease].uploadedData = uploadedData;
                    
                    // Display data immediately
                    displayData(parsedData);
                    predictBtn.style.display = 'block';
                    downloadBtn.style.display = 'none';
                    
                    // Show success message after data is displayed
                    Swal.fire({
                        icon: 'success',
                        title: 'File Uploaded Successfully!',
                        text: `Loaded ${parsedData.data.length} records.`,
                        confirmButtonColor: getThemeColor()
                    });
                }
                // If uploadedData is already set, it means the cleaning was handled in showValidationErrorsWithOption
            } else {
                // Validation failed or user cancelled
                uploadedData = null;
                predictedData = null;
                diseaseStates[currentDisease].uploadedData = null;
                diseaseStates[currentDisease].predictedData = null;
                
                // Show empty table again
                const config = diseaseConfigs[currentDisease];
                tableBody.innerHTML = '<tr><td colspan="' + (config.attributes.length + 1) + '" class="empty-table-message">No data uploaded yet. Upload a CSV file to see data here.</td></tr>';
            }
            
        } catch (error) {
            console.error('CSV parsing error:', error);
            
            // Reset data on error
            uploadedData = null;
            predictedData = null;
            diseaseStates[currentDisease].uploadedData = null;
            diseaseStates[currentDisease].predictedData = null;
            
            Swal.fire({
                icon: 'error',
                title: 'File Processing Error',
                html: `
                    <div style="text-align: left;">
                        <p><strong>Error:</strong> ${error.message}</p>
                        <p>Please ensure your file is a valid CSV with:</p>
                        <ul style="text-align: left; margin-left: 20px;">
                            <li>Headers in the first row</li>
                            <li>Data separated by commas</li>
                            <li>No empty rows</li>
                            <li>UTF-8 encoding</li>
                        </ul>
                    </div>
                `,
                confirmButtonColor: getThemeColor()
            });
        }
    };
    
    // reader.onerror = function() {
    //     // Reset data on file reading error
    //     uploadedData = null;
    //     predictedData = null;
        
    //     Swal.fire({
    //         icon: 'error',
    //         title: 'File Reading Error',
    //         text: 'Unable to read the selected file. Please try again.',
    //         confirmButtonColor: getThemeColor()
    //     });
    // };
    
    reader.readAsText(file);
}

    // Enhanced CSV parsing with better error handling
function parseCSV(csvData) {
    if (!csvData || typeof csvData !== 'string') {
        throw new Error('Invalid CSV data');
    }

    const lines = csvData.trim().split('\n');
    
    if (lines.length < 2) {
        throw new Error('CSV must have at least a header row and one data row');
    }

    // Parse headers - handle different separators and quotes
    const headerLine = lines[0];
    const headers = parseCSVLine(headerLine);
    
    if (headers.length === 0) {
        throw new Error('No headers found in CSV');
    }

    console.log('Parsed headers:', headers);

    const data = [];
    
    // Parse data rows
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line === '') continue; // Skip empty lines
        
        const values = parseCSVLine(line);
        
        if (values.length !== headers.length) {
            console.warn(`Row ${i} has ${values.length} values but expected ${headers.length}`);
            // Pad or trim values to match headers
            while (values.length < headers.length) {
                values.push('');
            }
            values.splice(headers.length);
        }
        
        const row = {};
        headers.forEach((header, index) => {
            row[header] = values[index] || '';
        });
        data.push(row);
    }

    if (data.length === 0) {
        throw new Error('No data rows found in CSV');
    }

    return { headers, data };
}

// Helper function to parse CSV line (handles quotes and commas)
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

    // Validate uploaded data with option to remove invalid records
function validateData(parsedData) {
    const config = diseaseConfigs[currentDisease];
    
    console.log('Expected headers:', config.attributes.map(attr => attr.id));
    console.log('CSV headers:', parsedData.headers);

    const expectedHeaders = config.attributes.map(attr => attr.id);
    
    // Define target/outcome columns that should be ignored
    const targetColumns = ['target', 'outcome', 'class', 'label', 'result'];
    
    // Filter out target columns from CSV headers for comparison
    const csvHeadersFiltered = parsedData.headers.filter(header => 
        !targetColumns.includes(header.toLowerCase())
    );
    
    console.log('CSV headers (filtered):', csvHeadersFiltered);

    // More flexible header matching
    const missingHeaders = [];
    const headerMap = {};
    
    expectedHeaders.forEach(expectedHeader => {
        let found = false;
        
        // Try exact match first
        if (csvHeadersFiltered.includes(expectedHeader)) {
            headerMap[expectedHeader] = expectedHeader;
            found = true;
        } else {
            // Try case-insensitive match
            const matchedHeader = csvHeadersFiltered.find(h => 
                h.toLowerCase() === expectedHeader.toLowerCase()
            );
            if (matchedHeader) {
                headerMap[expectedHeader] = matchedHeader;
                found = true;
            }
        }
        
        if (!found) {
            missingHeaders.push(expectedHeader);
        }
    });

    if (missingHeaders.length > 0) {
        console.log('Missing headers:', missingHeaders);
        
        // Show detailed error message for missing headers (this is still blocking)
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

    // Validate data values and collect invalid records
    const validationErrors = [];
    const invalidRowIndexes = new Set();
    
    for (let rowIndex = 0; rowIndex < parsedData.data.length; rowIndex++) {
        const row = parsedData.data[rowIndex];
        let hasErrors = false;
        
        for (const attr of config.attributes) {
            const actualHeader = headerMap[attr.id];
            const value = row[actualHeader] || row[attr.id] || row[attr.id.toLowerCase()] || 
                         row[attr.id.toUpperCase()] || row[attr.label];
            
            if (!validateAttributeValue(value, attr)) {
                validationErrors.push({
                    row: rowIndex + 1,
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
        
        // Show confirmation dialog asking if user wants to proceed by removing invalid records
        return showValidationErrorsWithOption(validationErrors, invalidRowIndexes, parsedData);
    }

    return true;
}

// Also update the showValidationErrorsWithOption function
async function showValidationErrorsWithOption(validationErrors, invalidRowIndexes, parsedData) {
    const totalRows = parsedData.data.length;
    const invalidCount = invalidRowIndexes.size;
    const validCount = totalRows - invalidCount;
    
    // Show detailed validation errors with option to proceed
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
    
    // Show confirmation dialog
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
        // Remove invalid records from the data
        const cleanedData = {
            headers: parsedData.headers,
            data: parsedData.data.filter((row, index) => !invalidRowIndexes.has(index))
        };
        
        // Update the global uploadedData with cleaned data
        uploadedData = cleanedData;

        // Save to state immediately
        diseaseStates[currentDisease].uploadedData = uploadedData;
        
        // Display the cleaned data immediately
        displayData(cleanedData);
        predictBtn.style.display = 'block';
        downloadBtn.style.display = 'none';
        
        // Show success message after data is displayed
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
        // User cancelled
        uploadedData = null;
        predictedData = null;
        diseaseStates[currentDisease].uploadedData = null;
        diseaseStates[currentDisease].predictedData = null;
        return false;
    }
}

// Also add a helper function to reset the file input
function resetFileInput() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.value = '';
    }
}

// Helper function to get expected value description
function getExpectedValueDescription(attr) {
    if (attr.type === 'number') {
        let desc = 'Number';
        if (attr.min !== undefined && attr.max !== undefined) {
            desc += ` (${attr.min}-${attr.max})`;
        } else if (attr.min !== undefined) {
            desc += ` (min: ${attr.min})`;
        } else if (attr.max !== undefined) {
            desc += ` (max: ${attr.max})`;
        }
        return desc;
    } else if (attr.type === 'categorical') {
        return `One of: ${attr.values.join(', ')}`;
    }
    return 'Valid value';
}

    // Validate individual attribute value
    function validateAttributeValue(value, attr) {
        if (value === undefined || value === null || value === '') {
            return false;
        }

        const numValue = parseFloat(value);

        if (attr.type === 'number') {
            if (isNaN(numValue)) return false;
            if (attr.min !== undefined && numValue < attr.min) return false;
            if (attr.max !== undefined && numValue > attr.max) return false;
        } else if (attr.type === 'categorical') {
            if (!attr.values.includes(numValue)) return false;
        }

        return true;
    }

    // Update the display data function to show record count in table
function displayData(data, showPrediction = false) {
    tableBody.innerHTML = '';
    
    // Show/hide prediction column
    document.querySelector('.prediction-column').style.display = showPrediction ? 'table-cell' : 'none';
    
    data.data.forEach((row, index) => {
        const tr = document.createElement('tr');
        
        // Add prediction cell (if showing predictions)
        if (showPrediction && predictedData) {
            const predictionCell = document.createElement('td');
            predictionCell.className = 'prediction-column';
            predictionCell.textContent = predictedData[index] || 'N/A';
            tr.appendChild(predictionCell);
        } else if (showPrediction) {
            const predictionCell = document.createElement('td');
            predictionCell.className = 'prediction-column';
            predictionCell.textContent = 'Processing...';
            tr.appendChild(predictionCell);
        }
        
        // Add data cells
        const config = diseaseConfigs[currentDisease];
        config.attributes.forEach(attr => {
            const td = document.createElement('td');
            const value = row[attr.id] || row[attr.id.toLowerCase()] || 
                         row[attr.id.toUpperCase()] || row[attr.label];
            td.textContent = value || 'N/A';
            tr.appendChild(td);
        });
        
        tableBody.appendChild(tr);
    });
    
    // Update empty table message to show record count
    if (data.data.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="' + (config.attributes.length + 1) + '" class="empty-table-message">No valid data found.</td></tr>';
    }
}

    // Show valid attribute values modal - Enhanced version
function showValidAttributeValues() {
    const config = diseaseConfigs[currentDisease];
    let content = `<div style="text-align: left; max-height: 400px; overflow-y: auto;">`;
    content += `<h3 style="margin-bottom: 15px; color: ${getThemeColor()};">Valid Attribute Values for ${config.name}</h3>`;
    
    config.attributes.forEach(attr => {
        content += `<div style="margin-bottom: 10px;">`;
        content += `<strong>${attr.label}:</strong> `;
        
        if (attr.type === 'number') {
            content += `Number (${attr.min !== undefined ? `Min: ${attr.min}` : 'No minimum'}${attr.max !== undefined ? `, Max: ${attr.max}` : ', No maximum'})`;
        } else if (attr.type === 'categorical') {
            // Check if values form a sequential range
            const isSequential = attr.values.every((val, idx) => 
                idx === 0 || val === attr.values[idx - 1] + 1
            );
            
            // Check if values and labels are the same (redundant)
            const isRedundant = attr.values.every((val, idx) => 
                val.toString() === attr.labels[idx]
            );
            
            if (isSequential && isRedundant) {
                // Show as range for sequential values
                content += `${attr.values[0]} to ${attr.values[attr.values.length - 1]} (scale)`;
            } else if (isRedundant) {
                // Just show the values without redundant labels
                content += `${attr.values.join(', ')}`;
            } else {
                // Show values with meaningful labels
                content += `${attr.values.map((val, idx) => `${val} (${attr.labels[idx]})`).join(', ')}`;
            }
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

    // Handle prediction
    async function handlePrediction() {
        if (!uploadedData) return;

        try {
            // Show loading state
            predictBtn.textContent = 'Processing...';
            predictBtn.disabled = true;
            document.querySelector('.table-container').classList.add('loading');

            const config = diseaseConfigs[currentDisease];
            
            // Prepare data for API
            const apiData = uploadedData.data.map(row => {
                const formattedRow = {};
                config.attributes.forEach(attr => {
                    const value = row[attr.id] || row[attr.id.toLowerCase()] || 
                                 row[attr.id.toUpperCase()] || row[attr.label];
                    formattedRow[attr.id] = parseFloat(value);
                });
                return formattedRow;
            });

            // Send to API
            const response = await fetch(`http://127.0.0.1:5000${config.endpoint}`, {
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
            
            // Process results
            predictedData = result.predictions.map(pred => 
                pred === 1 ? config.positiveClass : config.negativeClass
            );

            // Save to state
            diseaseStates[currentDisease].predictedData = predictedData;

            // Update display
            displayData(uploadedData, true);
            
            // Update UI
            predictBtn.style.display = 'none';
            downloadBtn.style.display = 'block';
            
            // Show success message
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
            // Reset button state
            predictBtn.textContent = 'Predict Class';
            predictBtn.disabled = false;
            document.querySelector('.table-container').classList.remove('loading');
        }
    }

    // Update the download function to include a note about cleaned data
function handleDownload() {
    if (!uploadedData || !predictedData) return;

    // Create CSV content
    const config = diseaseConfigs[currentDisease];
    const headers = ['Prediction', ...config.attributes.map(attr => attr.label)];
    let csvContent = headers.join(',') + '\n';

    uploadedData.data.forEach((row, index) => {
        const rowData = [predictedData[index]];
        config.attributes.forEach(attr => {
            const value = row[attr.id] || row[attr.id.toLowerCase()] || 
                         row[attr.id.toUpperCase()] || row[attr.label];
            rowData.push(value);
        });
        csvContent += rowData.join(',') + '\n';
    });

    // Create and trigger download
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentDisease}_predictions_cleaned.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    
    // Show download confirmation
    Swal.fire({
        icon: 'success',
        title: 'Download Complete!',
        text: 'The cleaned data with predictions has been downloaded.',
        confirmButtonColor: getThemeColor(),
        timer: 2000,
        showConfirmButton: false
    });
}
    // Get theme color for UI elements
    function getThemeColor() {
        const colors = {
            diabetes: '#00BF63',
            heart: '#DF6565',
            cancer: '#0097B2'
        };
        return colors[currentDisease];
    }

    // Initialize on page load
    initializePage();

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