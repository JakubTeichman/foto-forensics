import React from 'react';
import ReactECharts from 'echarts-for-react';

const ManipulationChart = () => {
    const option = {
        title: {
            text: 'Manipulation Chart'
        },
        tooltip: {},
        xAxis: {
            data: ['A', 'B', 'C', 'D', 'E']
        },
        yAxis: {},
        series: [{
            name: 'Example Data',
            type: 'bar',
            data: [5, 20, 36, 10, 10]
        }]
    };

    return (
        <div>
            <ReactECharts option={option} />
        </div>
    );
};

export default ManipulationChart;