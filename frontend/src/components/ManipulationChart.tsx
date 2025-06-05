import React, { useEffect } from 'react';
import * as echarts from 'echarts';

interface Props {
  manipulationScore: number;
}

const ManipulationChart: React.FC<Props> = ({ manipulationScore }) => {
  useEffect(() => {
    const chartDom = document.getElementById('manipulationChart');
    if (chartDom) {
      const myChart = echarts.init(chartDom);
      const option = {
        series: [
          {
            type: 'gauge',
            startAngle: 180,
            endAngle: 0,
            min: 0,
            max: 100,
            splitNumber: 10,
            itemStyle: { color: '#00e676' },
            progress: {
              show: true,
              width: 18,
              itemStyle: {
                color: {
                  type: 'linear',
                  x: 0, y: 0, x2: 1, y2: 0,
                  colorStops: [
                    { offset: 0, color: '#00e676' },
                    { offset: 1, color: '#00b8d4' }
                  ]
                }
              }
            },
            pointer: {
              icon: 'path://M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z',
              length: '60%',
              width: 8,
              offsetCenter: [0, '5%'],
              itemStyle: { color: '#00b8d4' }
            },
            axisLine: {
              lineStyle: {
                width: 18,
                color: [[1, 'rgba(0,184,212,0.2)']]
              }
            },
            axisTick: { show: false },
            splitLine: {
              length: 15,
              lineStyle: { width: 2, color: '#999' }
            },
            axisLabel: {
              distance: 25,
              color: '#999',
              fontSize: 12
            },
            title: {
              show: true,
              offsetCenter: [0, '30%'],
              fontSize: 14,
              color: '#fff'
            },
            detail: {
              valueAnimation: true,
              formatter: '{value}%',
              color: '#fff',
              fontSize: 24,
              offsetCenter: [0, '60%']
            },
            data: [
              {
                value: manipulationScore,
                name: 'Manipulation Score'
              }
            ],
            animation: false
          }
        ]
      };
      myChart.setOption(option);
    }
  }, [manipulationScore]);

  return <div id="manipulationChart" className="w-full h-64"></div>;
};

export default ManipulationChart;
