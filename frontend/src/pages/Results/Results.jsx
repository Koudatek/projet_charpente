import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { 
  Box, 
  Tabs, 
  Tab, 
  Typography, 
  Button, 
  Container,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import StructurePlot from './components/StructurePlot';
import EffortsPlot from './components/EffortsPlot';
import DeplacementsPlot from './components/DeplacementsPlot';
import { generatePdf } from '../../services/api';
import { saveAs } from 'file-saver';
import SaveIcon from '@mui/icons-material/Save';

// Composant pour le contenu des onglets
function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`results-tabpanel-${index}`}
      aria-labelledby={`results-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

// Composant pour l'onglet Graphique
const GraphiqueTab = ({ results }) => (
  <Box>
    <Typography variant="h6" gutterBottom>
      Visualisation de la structure
    </Typography>
    <StructurePlot 
      nodes={results.nodes} 
      bars={results.bars} 
      displacements={results.displacements} 
    />
  </Box>
);

// Composant pour l'onglet Données
const DonneesTab = ({ results }) => {
  const maxDisplacement = Math.max(
    ...Object.values(results.displacements || {}).map(([ux, uy]) => 
      Math.sqrt(ux**2 + uy**2)
    )
  );

  const maxForce = Math.max(
    ...Object.values(results.internal_forces || {}).map(Math.abs)
  );

  // Tableaux détaillés
  const nodes = results.nodes || [];
  const displacements = results.displacements || {};
  const bars = results.bars || [];
  const internal_forces = results.internal_forces || {};
  const reactions = results.reactions || {};

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Résultats numériques
      </Typography>
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle1" gutterBottom>
          Indicateurs clés
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          <Paper sx={{ p: 2, minWidth: 200 }}>
            <Typography variant="body2" color="text.secondary">
              Déplacement maximal
            </Typography>
            <Typography variant="h6">
              {(maxDisplacement * 1000).toFixed(3)} mm
            </Typography>
          </Paper>
          <Paper sx={{ p: 2, minWidth: 200 }}>
            <Typography variant="body2" color="text.secondary">
              Effort maximal
            </Typography>
            <Typography variant="h6">
              {(maxForce / 1000).toFixed(2)} kN
            </Typography>
          </Paper>
        </Box>
      </Box>

      {/* Tableau des déplacements nodaux */}
      <Typography variant="subtitle1" gutterBottom>
        Déplacements nodaux
      </Typography>
      <TableContainer component={Paper} sx={{ mb: 3 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Nœud</TableCell>
              <TableCell align="right">Ux (mm)</TableCell>
              <TableCell align="right">Uy (mm)</TableCell>
              <TableCell align="right">|U| (mm)</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {nodes.map(([label]) => {
              const [ux, uy] = displacements[label] || [0, 0];
              const mag = Math.sqrt(ux ** 2 + uy ** 2);
              return (
                <TableRow key={label}>
                  <TableCell>{label}</TableCell>
                  <TableCell align="right">{(ux * 1000).toFixed(3)}</TableCell>
                  <TableCell align="right">{(uy * 1000).toFixed(3)}</TableCell>
                  <TableCell align="right">{(mag * 1000).toFixed(3)}</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Tableau des efforts internes */}
      <Typography variant="subtitle1" gutterBottom>
        Efforts internes dans les barres
      </Typography>
      <TableContainer component={Paper} sx={{ mb: 3 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Barre</TableCell>
              <TableCell>Nœud 1</TableCell>
              <TableCell>Nœud 2</TableCell>
              <TableCell align="right">Effort (kN)</TableCell>
              <TableCell align="right">Type</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {bars.map(([id, n1, n2]) => {
              const force = internal_forces[id] || 0;
              return (
                <TableRow key={id}>
                  <TableCell>{id}</TableCell>
                  <TableCell>{n1}</TableCell>
                  <TableCell>{n2}</TableCell>
                  <TableCell align="right">{(force / 1000).toFixed(3)}</TableCell>
                  <TableCell align="right">{force > 0 ? 'Traction' : 'Compression'}</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Tableau des réactions d'appui si disponibles */}
      {reactions && Object.keys(reactions).length > 0 && (
        <>
          <Typography variant="subtitle1" gutterBottom>
            Réactions d'appui
          </Typography>
          <TableContainer component={Paper} sx={{ mb: 3 }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Appui</TableCell>
                  <TableCell align="right">Rx (N)</TableCell>
                  <TableCell align="right">Ry (N)</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(reactions).map(([label, [rx, ry]]) => (
                  <TableRow key={label}>
                    <TableCell>{label}</TableCell>
                    <TableCell align="right">{rx.toFixed(2)}</TableCell>
                    <TableCell align="right">{ry.toFixed(2)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </>
      )}
    </Box>
  );
};

// Composant pour l'onglet Export
const ExportTab = ({ results, params }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleExport = async () => {
    setLoading(true);
    setError('');
    try {
      // On utilise params (les valeurs du formulaire d'entrée)
      const paramsToSend = {
        ...params,
        L: Number(params.L),
        H: Number(params.H),
        m: Number(params.m),
        alpha_deg: Number(params.alpha_deg),
        F: Number(params.F)
      };
      const pdfBlob = await generatePdf(paramsToSend);
      // Création d'un lien de téléchargement
      const url = window.URL.createObjectURL(new Blob([pdfBlob], { type: 'application/pdf' }));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `rapport_${params.project_name || 'charpente'}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError("Erreur lors de l'export PDF : " + (err.message || 'inconnue'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Export des résultats
      </Typography>
      <Typography variant="body1" paragraph>
        Téléchargez un rapport PDF complet contenant tous les résultats de l'analyse.
      </Typography>
      {error && (
        <Typography color="error" sx={{ mb: 2 }}>{error}</Typography>
      )}
      <Button 
        variant="contained" 
        color="primary" 
        size="large"
        onClick={handleExport}
        disabled={loading}
      >
        {loading ? 'Export en cours...' : 'Exporter en PDF'}
      </Button>
    </Box>
  );
};

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  // On attend maintenant { results, params } dans location.state
  const { results, params } = location.state || {};
  const [tabValue, setTabValue] = useState(0);

  if (!results) {
    return (
      <Container maxWidth="md" sx={{ mt: { xs: 2, sm: 4 } }}>
        <Paper sx={{ p: { xs: 2, sm: 3 }, textAlign: 'center' }}>
          <Typography variant="h6" gutterBottom sx={{ fontSize: { xs: '1.1rem', sm: '1.5rem' } }}>
            Aucun résultat à afficher
          </Typography>
          <Typography variant="body1" paragraph>
            Veuillez d'abord lancer un calcul.
          </Typography>
          <Button 
            variant="contained" 
            onClick={() => navigate('/new-project')}
            sx={{ width: { xs: '100%', sm: 'auto' } }}
          >
            Nouveau projet
          </Button>
        </Paper>
      </Container>
    );
  }

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Fonction d'export JSON des paramètres du projet
  const handleExportProject = () => {
    if (!params) return;
    const blob = new Blob([JSON.stringify(params, null, 2)], { type: 'application/json' });
    saveAs(blob, `${params.project_name || 'projet_charpente'}.json`);
  };

  return (
    <Container maxWidth="lg" sx={{ mt: { xs: 2, sm: 4 }, px: { xs: 0.5, sm: 2 } }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: 1, borderColor: 'divider', mb: 2 }}>
        <Box>
          <Typography variant="h4" gutterBottom sx={{ fontSize: { xs: '1.5rem', sm: '2.2rem' } }}>
            Résultats de l'analyse
          </Typography>
          <Typography variant="body1" color="text.secondary" gutterBottom sx={{ fontSize: { xs: '1rem', sm: '1.2rem' } }}>
            Projet: {params?.project_name || 'Non spécifié'}
          </Typography>
        </Box>
        <Button
          variant="text"
          color="primary"
          startIcon={<SaveIcon />}
          onClick={handleExportProject}
          sx={{ minWidth: 0, p: 1, fontSize: { xs: '0.9rem', sm: '1rem' } }}
          aria-label="Enregistrer le projet"
        >
          <Box sx={{ display: { xs: 'none', sm: 'inline' } }}>Enregistrer le projet</Box>
        </Button>
      </Box>

      <Paper sx={{ width: '100%', p: { xs: 1, sm: 2 } }}>
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange} 
          aria-label="results tabs"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="Graphique" />
          <Tab label="Efforts internes" />
          <Tab label="Déplacements" />
          <Tab label="Données" />
          <Tab label="Export" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <Box sx={{ width: '100%', overflowX: 'auto', minHeight: 300 }}>
            <StructurePlot 
              nodes={results.nodes} 
              bars={results.bars} 
              displacements={results.displacements} 
              style={{ width: '100%', maxWidth: '100%' }}
            />
          </Box>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Box sx={{ width: '100%', overflowX: 'auto', minHeight: 300 }}>
            <EffortsPlot results={results} style={{ width: '100%', maxWidth: '100%' }} />
          </Box>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Box sx={{ width: '100%', overflowX: 'auto', minHeight: 300 }}>
            <DeplacementsPlot results={results} style={{ width: '100%', maxWidth: '100%' }} />
          </Box>
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <Box sx={{ width: '100%', overflowX: 'auto' }}>
            <DonneesTab results={results} />
          </Box>
        </TabPanel>

        <TabPanel value={tabValue} index={4}>
          <ExportTab results={results} params={params} />
        </TabPanel>
      </Paper>

      <Box sx={{ mt: 2, textAlign: 'center' }}>
        <Button 
          variant="outlined" 
          onClick={() => navigate('/new-project')}
          sx={{ mr: 1, width: { xs: '100%', sm: 'auto' }, mb: { xs: 1, sm: 0 } }}
        >
          Nouveau calcul
        </Button>
        <Button 
          variant="outlined" 
          onClick={() => navigate('/')}
          sx={{ width: { xs: '100%', sm: 'auto' } }}
        >
          Accueil
        </Button>
        {/* Bouton d'export supprimé ici */}
      </Box>
    </Container>
  );
};

export default Results; 