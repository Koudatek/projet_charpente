import React from 'react';
import Plot from 'react-plotly.js';

const EffortsPlot = ({ results }) => {
  const { nodes = [], bars = [], internal_forces = {} } = results || {};
  const nodeMap = {};
  nodes.forEach(([label, x, y]) => {
    nodeMap[label] = { x, y };
  });

  // Traces des barres colorées selon l'effort
  const barTraces = bars.map(([id, n1, n2]) => {
    const node1 = nodeMap[n1];
    const node2 = nodeMap[n2];
    if (!node1 || !node2) return null;
    const force = internal_forces[id] || 0;
    const isTraction = force > 0;
    return {
      x: [node1.x, node2.x],
      y: [node1.y, node2.y],
      mode: 'lines',
      type: 'scatter',
      line: {
        color: isTraction ? 'red' : 'blue',
        width: 4
      },
      hoverinfo: 'text',
      text: `${id}: ${isTraction ? 'Traction' : 'Compression'}\n${(force/1000).toFixed(2)} kN`,
      showlegend: false,
    };
  }).filter(Boolean);

  // Traces des nœuds
  const nodeLabels = {
    x: nodes.map(n => n[1]),
    y: nodes.map(n => n[2]),
    text: nodes.map(n => n[0]),
    mode: 'text',
    type: 'scatter',
    textposition: 'top center',
    showlegend: false,
    hoverinfo: 'text',
    textfont: { color: 'black', size: 12, family: 'Arial' },
  };

  return (
    <Plot
      data={[...barTraces, nodeLabels]}
      layout={{
        title: 'Efforts internes dans les barres',
        xaxis: { title: 'X (m)', zeroline: false },
        yaxis: { title: 'Y (m)', zeroline: false, scaleanchor: 'x', scaleratio: 1 },
        showlegend: false,
        width: 700,
        height: 500,
        margin: { t: 40, l: 40, r: 40, b: 40 },
        annotations: [
          {
            xref: 'paper', yref: 'paper',
            x: 1.05, y: 1,
            text: '<span style="color:red">Traction</span>',
            showarrow: false,
            font: { size: 14 }
          },
          {
            xref: 'paper', yref: 'paper',
            x: 1.05, y: 0.95,
            text: '<span style="color:blue">Compression</span>',
            showarrow: false,
            font: { size: 14 }
          }
        ]
      }}
      config={{ responsive: true }}
    />
  );
};

export default EffortsPlot; 