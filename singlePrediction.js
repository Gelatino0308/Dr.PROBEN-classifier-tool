document.addEventListener('DOMContentLoaded', () => {
    
    const form = document.getElementById('predictionForm');
    const displayResult = document.getElementById('result');
    let percentText = document.getElementById('percentText');
    let percentage = 0;

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
                // Disable the initial appearance animation
                duration: 0,
                // Keep the smooth data transition
                animateRotate: true,
                animateScale: false
            },
            plugins: {
                legend: { display: true },
                tooltip: { enabled: true },
            },
            layout: {
                padding: 5
            }
        },
        plugins: [shadowPlugin]
    };

    const chart = new Chart(document.getElementById('doughnutChart'), config);

    // Handle form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Get form data
        const formData = {
            pregnancies: document.getElementById('pregnancies').value,
            plasma: document.getElementById('plasma').value,
            BP: document.getElementById('BP').value,
            skin: document.getElementById('skin').value,
            insulin: document.getElementById('insulin').value,
            BMI: document.getElementById('BMI').value,
            pedigree: document.getElementById('pedigree').value,
            age: document.getElementById('age').value
        };

        try {
            // Show loading state
            percentText.textContent = 'Loading...';
            
            // Send data to API
            const response = await fetch('http://127.0.0.1:5000/api/predict/diabetes', {
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
            
            // Update the prediction text
            document.querySelector('.result-section h1').textContent = 
                result.prediction === 1 ? 'DIABETIC' : 'NON-DIABETIC';
            
            // Update the description based on the prediction
            const description = document.querySelector('.result-section p');
            if (result.prediction === 1) {
                description.textContent = "Diabetic means the person has diabetes, a chronic disease that affects how your body turns food into energy. It occurs when your pancreas doesn't make enough insulin or your cells don't respond to insulin properly.";
            } else {
                description.textContent = 'Non-diabetic means the absence of diabetes. Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood glucose.';
            }

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
        displayResult.style.display = 'none';
    });
});
