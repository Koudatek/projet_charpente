import React from 'react';
import Plot from 'react-plotly.js';

/**
 * @param {Object[]} nodes - Array of [label, x, y]
 * @param {Object[]} bars - Array of [id, label1, label2]
 * @param {Object} displacements - { label: [ux, uy] }
 */
const StructurePlot = ({ nodes = [], bars = [], displacements = {} }) => {
  // Créer un mapping label -> coordonnées
  const nodeMap = {};
  nodes.forEach(([label, x, y]) => {
    nodeMap[label] = { x, y };
  });

  // Structure initiale (bleu)
  const barTraces = bars.map(([id, n1, n2]) => {
    const node1 = nodeMap[n1];
    const node2 = nodeMap[n2];
    if (!node1 || !node2) return null;
    return {
      x: [node1.x, node2.x],
      y: [node1.y, node2.y],
      mode: 'lines+markers',
      type: 'scatter',
      line: { color: 'blue', width: 2 },
      marker: { size: 6, color: 'black' },
      hoverinfo: 'none',
      showlegend: false,
      name: 'Initiale',
    };
  }).filter(Boolean);

  // Structure déformée (rouge)
  const scale = 100; // Facteur d'amplification visuelle
  const barTracesDeformed = bars.map(([id, n1, n2]) => {
    const node1 = nodeMap[n1];
    const node2 = nodeMap[n2];
    if (!node1 || !node2) return null;
    const d1 = displacements[n1] || [0, 0];
    const d2 = displacements[n2] || [0, 0];
    return {
      x: [node1.x + d1[0] * scale, node2.x + d2[0] * scale],
      y: [node1.y + d1[1] * scale, node2.y + d2[1] * scale],
      mode: 'lines+markers',
      type: 'scatter',
      line: { color: 'red', width: 2, dash: 'dot' },
      marker: { size: 6, color: 'red' },
      hoverinfo: 'none',
      showlegend: false,
      name: 'Déformée',
    };
  }).filter(Boolean);

  // Traces des nœuds (pour affichage des labels)
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
        data={[...barTraces, ...barTracesDeformed, nodeLabels]}
        layout={{
          title: 'Structure initiale (bleu) et déformée (rouge, amplifiée)',
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
              text: '<span style="color:blue">Bleu</span> : structure initiale',
              showarrow: false,
              font: { size: 14 }
            },
            {
              xref: 'paper', yref: 'paper',
              x: 1.05, y: 0.95,
              text: '<span style="color:red">Rouge pointillé</span> : structure déformée (amplifiée)',
              showarrow: false,
              font: { size: 14 }
            }
          ]
        }}
        config={{ responsive: true }}
      />
      <div style={{ marginTop: 8, fontSize: 14, color: '#555' }}>
        <b>Légende :</b> <span style={{ color: 'blue' }}>Bleu</span> = structure initiale,&nbsp;
        <span style={{ color: 'red' }}>Rouge pointillé</span> = structure déformée (amplifiée ×100)
      </div>
    </>
  );
};

export default StructurePlot; 