// src/components/MetadataChart.tsx
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

    const hasExif = exifCount > 0;
    const hasGps = gpsCount > 0;

    const data: { value: number; name: string }[] = [];
    if (hasExif) data.push({ value: exifCount, name: 'EXIF Attributes' });
    if (hasGps) data.push({ value: gpsCount, name: 'GPS Attributes' });
    if (!hasExif && !hasGps) data.push({ value: 1, name: 'No Metadata' });

    const option = {
      backgroundColor: 'transparent',
      title: {
        text: 'Metadata Attribute Distribution',
        left: 'center',
        top: 5, // 📏 bliżej górnej krawędzi kontenera
        textStyle: {
          color: '#2dd4bf',
          fontSize: 18,
          fontWeight: '600',
        },
      },
      tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      series: [
        {
          name: 'Metadata Attributes',
          type: 'pie',
          radius: ['45%', '70%'],
          top: 40, // 📏 dodany odstęp od tytułu
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
          data,
        },
      ],
    };

    myChart.setOption(option as any);
    return () => myChart.dispose();
  }, [exifCount, gpsCount]);

  return (
    <div className="mt-6 mb-8">
      <div id="metadataChart" className="w-full h-64" />
    </div>
  );
};

export default MetadataChart;
