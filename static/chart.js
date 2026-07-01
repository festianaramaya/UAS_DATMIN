document.addEventListener("DOMContentLoaded", function () {

    const canvas = document.getElementById("emotionChart");

    if (!canvas) return;

    const labels = JSON.parse(canvas.dataset.labels);

    const values = JSON.parse(canvas.dataset.values);

    new Chart(canvas, {

        type: "bar",

        data: {

            labels: labels,

            datasets: [

                {

                    label: "Probabilitas (%)",

                    data: values,

                    borderWidth: 1,

                    borderRadius: 8,

                    backgroundColor: [

                        "#4F46E5",

                        "#06B6D4",

                        "#22C55E",

                        "#F59E0B",

                        "#EF4444",

                        "#EC4899"

                    ]

                }

            ]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            plugins: {

                legend: {

                    display: false

                }

            },

            scales: {

                y: {

                    beginAtZero: true,

                    max: 100

                }

            }

        }

    });

});