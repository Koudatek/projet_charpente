import React from 'react';
import Plot from 'react-plotly.js';

const DeplacementsPlot = ({ results }) => {
  const { nodes = [], bars = [], displacements = {} } = results || {};
  const nodeMap = {};
  nodes.forEach(([label, x, y]) => {
    nodeMap[label] = { x, y };
  });
  const scale = 100; // facteur d'amplification visuelle

  // Traces des barres (structure initiale)
  const barTraces = bars.map(([id, n1, n2]) => {
    const node1 = nodeMap[n1];
    const node2 = nodeMap[n2];
    if (!node1 || !node2) return null;
    return {
      x: [node1.x, node2.x],
      y: [node1.y, node2.y],
      mode: 'lines',
      type: 'scatter',
      line: { color: 'grey', width: 2 },
      hoverinfo: 'none',
      showlegend: false,
    };
  }).filter(Boolean);

  // Traces des flèches de déplacement
  const arrowTraces = nodes.map(([label, x, y]) => {
    const [ux, uy] = displacements[label] || [0, 0];
    return {
      x: [x, x + ux * scale],
      y: [y, y + uy * scale],
      mode: 'lines+markers',
      type: 'scatter',
      line: { color: 'red', width: 3, dash: 'dot' },
      marker: { size: 6, color: 'red' },
      hoverinfo: 'text',
      text: `${label}: (Ux=${(ux*1000).toFixed(2)} mm, Uy=${(uy*1000).toFixed(2)} mm)`,
      showlegend: false,
    };
  });

  // Traces des nœuds (labels)
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
    <>
      <Plot
        data={[...barTraces, ...arrowTraces, nodeLabels]}
        layout={{
          title: 'Déplacements nodaux (flèches amplifiées)',
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
              text: '<span style="color:red">Flèches rouges</span> : déplacement nodal (amplifié ×100, valeurs en mm)',
              showarrow: false,
              font: { size: 14 }
            }
          ]
        }}
        config={{ responsive: true }}
      />
      <div style={{ marginTop: 8, fontSize: 14, color: '#555' }}>
        <b>Légende :</b> <span style={{ color: 'red' }}>Flèches rouges</span> = déplacement nodal (amplifié ×100, valeurs en mm)
      </div>
    </>
  );
};

export default DeplacementsPlot; 