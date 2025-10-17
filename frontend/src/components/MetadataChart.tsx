import React, { useEffect } from 'react';
import * as echarts from 'echarts';

interface Props {
  resolution: string;
  fileSize: string;
  hasExif: boolean;
  hasGps: boolean;
}

const MetadataChart: React.FC<Props> = ({ resolution, fileSize, hasExif, hasGps }) => {
  useEffect(() => {
    const chartDom = document.getElementById('metadataChart');
    if (!chartDom) return;
    const myChart = echarts.init(chartDom);

    const option: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      title: {
        text: 'Image Metadata Summary',
        left: 'center',
        top: 10,
        textStyle: { color: '#00e676', fontSize: 16 },
      },
      tooltip: {
        trigger: 'item',
        formatter: '{b}: {c} ({d}%)',
      },
      series: [
        {
          name: 'Metadata',
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
            formatter: '{b}',
            color: '#fff',
            fontSize: 12,
          },
          data: [
            { value: hasExif ? 1 : 0, name: hasExif ? 'EXIF Present' : 'No EXIF' },
            { value: hasGps ? 1 : 0, name: hasGps ? 'GPS Present' : 'No GPS' },
          ],
        },
      ],
    };

    myChart.setOption(option as any);


    return () => {
      myChart.dispose();
    };
  }, [resolution, fileSize, hasExif, hasGps]);

  return <div id="metadataChart" className="w-full h-64"></div>;
};

export default MetadataChart;
