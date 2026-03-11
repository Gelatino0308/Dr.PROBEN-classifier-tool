/**
 * Disease Configuration Module
 * 
 * Contains all disease-specific configurations, state structures, and informational content
 * for the Dr. PROBEN prediction interface.
 * 
 * Exports:
 * - diseaseStates: Initial state structure for each disease
 * - diseaseConfigs: Attribute definitions, endpoints, and class information
 * - diseaseInfoContent: Modal content for disease information displays
 */

// State management for both modes
export const diseaseStates = {
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
export const diseaseConfigs = {
    diabetes: {
        name: 'Diabetes',
        endpoint: '/api/predict/diabetes',
        batchEndpoint: '/api/predict/diabetes/batch',
        positiveClass: 'DIABETIC',
        negativeClass: 'NON-DIABETIC',
        positiveDesc: "Diabetic means the person has diabetes, a chronic disease that affects how your body turns food into energy. It occurs when your pancreas doesn't make enough insulin or your cells don't respond to insulin properly.",
        negativeDesc: 'Non-diabetic means the absence of diabetes. Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood glucose.',
        attributes: [
            { id: 'Number of Pregnancies', label: 'Number of Pregnancies', placeholder: '0', min: '0', max: 20, type: 'number', 
                info: 'If you have been pregnant twice, you would enter "2." If you have never been pregnant, you would enter "0."' 
            },
            { id: 'Plasma Glucose Concentration', label: 'Plasma Glucose Concentration', placeholder: '0 (mg/dL)', min: '0', max: 300, type: 'number', 
                info: 'This measures the amount of sugar in your blood. You will need to get this value from a recent blood test, often called a blood sugar test or glucose test. Look for a result listed as "Fasting Plasma Glucose" or similar, which is measured in milligrams per deciliter (mg/dL).'
            },
            { id: 'Diastolic Blood Pressure', label: 'Diastolic Blood Pressure', placeholder: '0 (mm Hg)', min: '0', max: 200, type: 'number',
                info: 'This is the second, or lower, number in a blood pressure reading. A reading is typically written as two numbers, like "120/80." In this example, "80" is the diastolic pressure. You can get this from a recent doctor\'s visit or a home blood pressure monitor.'
            },
            { id: 'Triceps Skin Fold Thickness', label: 'Triceps Skin Fold Thickness', placeholder: '0 (mm)', min: '0', max: 100, type: 'number',
                info: 'This value is a way to estimate the amount of body fat by measuring the thickness of a fold of skin and fat on the back of your upper arm. This measurement is usually taken with a special tool called a caliper. You will need to get this value from your doctor.'    
            },
            { id: '2-Hour Serum Insulin', label: '2-Hour Serum Insulin', placeholder: '0 (µU/mL)', min: '0', max: 1000, type: 'number',
                info: 'This measures the amount of insulin in your blood specifically two hours after you\'ve taken a glucose tolerance test. It shows how well your body processes sugar over time. This value should be obtained from a specific blood test.'
            },
            { id: 'Body Mass Index', label: 'Body Mass Index', placeholder: '0.0 (kg/m²)', min: '0', max: 70, type: 'number', step: 'any',
                info: 'Your BMI is a value calculated from your weight and height that helps determine if you are at a healthy weight. To find your BMI, you can use an online calculator. Simply enter your height and weight, and the calculator will provide your BMI value. For example, if you weigh 150 lbs and are 5\'5" tall, your BMI is approximately 25.'
            },
            { id: 'Diabetes Pedigree Function', label: 'Diabetes Pedigree Function', placeholder: '0.000', min: '0', max: 5, type: 'number', step: 'any',
                info: 'This is a complex score that quantifies the genetic risk of diabetes based on your family history. You won\'t have a number for this yourself. This value is typically calculated by the diagnostic tool based on the family history information you provide, such as whether your parents or siblings have diabetes.'
            },
            { id: 'Age', label: 'Age', placeholder: '0', min: '0', max: 120, type: 'number',
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
            { id: 'Age', label: 'Age', placeholder: '0', min: '0', max: 120, type: 'number',
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
            { id: 'Resting Blood Pressure', label: 'Resting Blood Pressure', placeholder: '0 (mm Hg)', min: '0', max: 300, type: 'number',
                info: 'This is the top number of your blood pressure reading, measured while you are at rest. It is measured in millimeters of mercury (mm Hg).'
            },
            { id: 'Serum Cholesterol', label: 'Serum Cholesterol', placeholder: '0 (mg/dL)', min: '0', max: 1000, type: 'number',
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
            { id: 'Maximum Heart Rate', label: 'Maximum Heart Rate', placeholder: '0', min: '0', max: 250, type: 'number',
                info:'This is the highest heart rate you reached during a stress or exercise test. This measurement is often taken on a treadmill or stationary bike while your heart rate is monitored.'
            },
            { id: 'Exercise Induced Angina', label: 'Exercise Induced Angina', type: 'radio', 
                options: [
                    { value: '1', label: 'Yes' },
                    { value: '0', label: 'No' }
                ],
                info: 'This indicates whether you experienced chest pain during physical exercise.\n• Yes: You experienced chest pain during exercise.\n• No: You did not experience chest pain during exercise.'
            },
            { id: 'ST Depression (Oldpeak)', label: 'ST Depression (Oldpeak)', placeholder: '0.0', min: '0', max: 10, type: 'number', step: 'any',
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
            { id: 'Number of Major Vessels', label: 'Number of Major Vessels', min: '0', max: '3', type: 'radio',
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
            { id: 'Clump Thickness', label: 'Clump Thickness', min: '1', max: '10', type: 'radio', 
                info:'Refers to the degree to which cells are clustered together. Higher thickness values may indicate abnormal cell growth or potential malignancy.' 
            },
            { id: 'Uniformity of Cell Size', label: 'Uniformity of Cell Size', min: '1', max: '10', type: 'radio', 
                info:'Measures the consistency in cell sizes within the sample. Significant variation in size may suggest the presence of abnormal or cancerous cells.' 
            },
            { id: 'Uniformity of Cell Shape', label: 'Uniformity of Cell Shape', min: '1', max: '10', type: 'radio', 
                info:'Evaluates the uniformity of cell shapes. Normal cells generally maintain consistent shapes, while irregular shapes may be indicative of malignancy.' 
            },
            { id: 'Marginal Adhesion', label: 'Marginal Adhesion', min: '1', max: '10', type: 'radio', 
                info:'Describes the extent to which cells adhere to one another. Poor adhesion may signify abnormal or invasive cellular behavior.' 
            },
            { id: 'Single Epithelial Cell Size', label: 'Single Epithelial Cell Size', min: '1', max: '10', type: 'radio', 
                info:'Represents the average size of individual epithelial cells. Enlarged epithelial cells are often associated with abnormal cellular activity.' 
            },
            { id: 'Bare Nuclei', label: 'Bare Nuclei', min: '1', max: '10', type: 'radio', 
                info:'Indicates the presence of nuclei without surrounding cytoplasm. A higher count of bare nuclei is commonly observed in malignant samples.' 
            },
            { id: 'Bland Chromatin', label: 'Bland Chromatin', min: '1', max: '10', type: 'radio', 
                info:'Refers to the texture and appearance of the chromatin within the nucleus. Coarse or uneven chromatin patterns may suggest abnormal or cancerous growth.' 
            },
            { id: 'Normal Nucleoli', label: 'Normal Nucleoli', min: '1', max: '10', type: 'radio', 
                info:'Pertains to the visibility and prominence of nucleoli within the nucleus. Prominent or multiple nucleoli are often linked to increased cellular activity, typical of cancerous cells.' 
            },
            { id: 'Mitoses', label: 'Mitoses', min: '1', max: '10', type: 'radio', 
                info:'Measures the frequency of cell division. An elevated mitotic rate reflects rapid cellular proliferation, which may indicate malignant behavior.' 
            }
        ]
    }
};

// Disease information content for modals
export const diseaseInfoContent = {
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