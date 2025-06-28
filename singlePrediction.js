const percentage = Math.floor(Math.random() * 101);
document.getElementById('percentText').textContent = percentage + '%';

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
        plugins: {
            legend: { display: false },
            tooltip: { enabled: false },
        },
        layout: {
            padding: 5
        }
    },
    plugins: [shadowPlugin]
};

new Chart(document.getElementById('doughnutChart'), config);