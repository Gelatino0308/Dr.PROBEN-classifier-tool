/**
 * Navigation and landing modal logic for Dr. PROBEN
 * Handles disease selection and disclaimer modal
 */

let pendingDisease = null;

function navigateToDisease(disease) {
    sessionStorage.removeItem('selectedMode');
    
    const skipModal = sessionStorage.getItem('drproben_skip_landing') === '1';
    if (skipModal) {
        sessionStorage.setItem('selectedDisease', disease);
        window.location.href = 'views/prediction.html';
        return;
    }
    openLandingModal(disease);
}

function openLandingModal(disease) {
    pendingDisease = disease;

    Swal.fire({
        html: `
            <div class="disclaimer-modal-content">
                <div class="modal-icon-container">
                    <img src="assets/warning.svg">
                    <h2 class="modal-title-text">Disclaimer!</h2>
                </div>
                <div class="modal-description-text">
                    <p>This tool serves as a guide for the decision-making of medical professionals such as doctors and specialists. This is <strong>NOT A DIAGNOSIS</strong>. Do not treat it as such.</p>
                    <p>Dr. PROBEN's outputs should not be considered, interpreted, or used as a substitute for professional medical advice or diagnosis by a qualified healthcare provider.</p>
                </div>
                <div class="modal-checkbox-container">
                    <input type="checkbox" id="modal-dont-show" class="modal-checkbox">
                    <label for="modal-dont-show" class="modal-checkbox-label">Don't show this again</label>
                </div>
            </div>
        `,
        showCancelButton: false,
        showConfirmButton: true,
        confirmButtonText: 'Yes, I understand',
        confirmButtonColor: '#00BF63',
        allowOutsideClick: true,
        allowEscapeKey: true,
        customClass: {
            popup: 'disclaimer-popup',
            confirmButton: 'disclaimer-confirm-btn'
        },
        width: '600px',
        padding: '30px',
        buttonsStyling: true,
        didOpen: () => {
            const confirmBtn = document.querySelector('.disclaimer-confirm-btn');
            if (confirmBtn) confirmBtn.focus();
        }
    }).then((result) => {
        if (result.isConfirmed) {
            proceedFromLandingModal();
        }
    });
}

/**
 * Handle modal confirmation and navigate to prediction page
 * Stores user preference for skipping modal in future sessions
 */
function proceedFromLandingModal() {

    const dontShow = document.getElementById('modal-dont-show')?.checked;
    if (dontShow) {
        sessionStorage.setItem('drproben_skip_landing', '1');
    }
    
    sessionStorage.setItem('selectedDisease', pendingDisease);
    window.location.href = 'views/prediction.html';
}