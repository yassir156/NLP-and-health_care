

function test(){
    console.log(amin);
}

  
var options = {
    series: [{
    name: 'Number',
    data: [1,8,3,2,2,2,2,1]
  }],
    chart: {
    height: 350,
    type: 'bar',
  },
  plotOptions: {
    bar: {
      borderRadius: 10,
      dataLabels: {
        position: 'top', // top, center, bottom
      },
    }
  },
  dataLabels: {
    enabled: true,
    formatter: function (val) {
      return val;
    },
    offsetY: -20,
    style: {
      fontSize: '12px',
      colors: ["#304758"]
    }
  },
  
  xaxis: {
    categories: ['cluster_0','cluster_1','cluster_2','cluster_3','cluster_4','cluster_5','cluster_6','cluster_7'],
    position: 'top',
    axisBorder: {
      show: false
    },
    axisTicks: {
      show: false
    },
    crosshairs: {
      fill: {
        type: 'gradient',
        gradient: {
          colorFrom: '#D8E3F0',
          colorTo: '#BED1E6',
          stops: [0, 100],
          opacityFrom: 0.4,
          opacityTo: 0.5,
        }
      }
    },
    tooltip: {
      enabled: true,
    }
  },
  yaxis: {
    axisBorder: {
      show: false
    },
    axisTicks: {
      show: false,
    },
    labels: {
      show: false,
      formatter: function (val) {
        return val + "%";
      }
    }
  
  },
  title: {
    text: 'Number of Pdfs in each Cluster',
    floating: true,
    offsetY: 330,
    align: 'center',
    style: {
      color: '#444'
    }
  }
  };
  var colors = ['#008FFB', '#00E396', '#FEB019', '#FF4560', '#775DD0', '#546E7A', '#26a69a', '#D10CE8'];
  var options_2 = {
    series: [{
    data: [6, 3, 1, 5, 1, 1, 1, 2]
  }],
    chart: {
    height: 350,
    type: 'bar',
    events: {
      click: function(chart, w, e) {
        // console.log(chart, w, e)
      }
    }
  },
  colors: colors,
  plotOptions: {
    bar: {
      columnWidth: '45%',
      distributed: true,
    }
  },
  dataLabels: {
    enabled: false
  },
  legend: {
    show: false
  },
  xaxis: {
    categories: [
      [' Elsevier', ''],
      ['Springer', 'Smith'],
      ['Arbortext Advanced', 'Print Publisher'],
      'Adobe InDesign',
      ['LaTeX with', 'hyperref package'],
      ['FrameMaker', ''],
      ['3B2 Total', 'Publishing System'],
      ['Apex CoVantage', ''], 
    ],
    labels: {
      style: {
        colors: colors,
        fontSize: '12px'
      }
    }
  }
  };

  var chart = new ApexCharts(document.querySelector("#chart"), options);
  chart.render();
  var chart_2 = new ApexCharts(document.querySelector("#chart_2"), options_2);
  chart_2.render();
  var chart_3 = new ApexCharts(document.querySelector("#chart_3"), options);
  chart_3.render();
  var chart_4 = new ApexCharts(document.querySelector("#chart_4"), options);
  chart_4.render