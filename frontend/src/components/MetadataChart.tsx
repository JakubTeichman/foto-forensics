import React, { useEffect } from 'react';
import * as echarts from 'echarts';

interface Props {
  exifCount: number;
  gpsCount: number;
}

const MetadataChart: React.FC<Props> = ({ exifCount, gpsCount }) => {
  useEffect(() => {
    const chartDom = document.getElementById('metadataChart');
    if (!chartDom) return;
    const myChart = echarts.init(chartDom);

    const total = exifCount + gpsCount;
    const option: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      title: {
        text: 'Metadata Attribute Distribution',
        left: 'center',
        top: 10,
        textStyle: { color: '#00e676', fontSize: 16 },
      },
      tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      series: [
        {
          name: 'Metadata Attributes',
          type: 'pie',
          radius: ['45%', '70%'],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 8,
            borderColor: '#000',
            borderWidth: 2,
          },
          label: {
            show: true,
            formatter: '{b} ({c})',
            color: '#fff',
            fontSize: 12,
          },
          data: [
            { value: exifCount, name: 'EXIF Attributes' },
            { value: gpsCount, name: 'GPS Attributes' },
            { value: total === 0 ? 1 : 0, name: total === 0 ? 'No Metadata' : '' },
          ].filter((d) => d.name !== ''),
        },
      ],
    };

    myChart.setOption(option as any);
    return () => myChart.dispose();
  }, [exifCount, gpsCount]);

  return <div id="metadataChart" className="w-full h-64"></div>;
};

export default MetadataChart;
