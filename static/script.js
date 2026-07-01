// ===============================
// Loading Button
// ===============================

document.addEventListener("DOMContentLoaded", () => {

    const form = document.querySelector("form");
    const button = document.querySelector("button[type='submit']");

    if (form && button) {

        form.addEventListener("submit", () => {

            button.disabled = true;

            button.innerHTML = `
                <span class="spinner-border spinner-border-sm"></span>
                Menganalisis...
            `;

        });

    }

});

// ===============================
// Auto Resize Textarea
// ===============================

const textarea = document.querySelector("textarea");

if (textarea) {

    textarea.addEventListener("input", function () {

        this.style.height = "auto";

        this.style.height = this.scrollHeight + "px";

    });

}

// ===============================
// Smooth Scroll
// ===============================

window.onload = function () {

    const result = document.getElementById("hasil");

    if (result) {

        result.scrollIntoView({

            behavior: "smooth"

        });

    }

};
// ===============================
// Progress Bar
// ===============================

document.addEventListener("DOMContentLoaded", () => {

    const progress = document.querySelector(".progress-confidence");

    if (progress) {

        const confidence = progress.dataset.confidence;

        progress.style.width = confidence + "%";

    }

});