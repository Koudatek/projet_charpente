import React, { useState, useEffect } from 'react';
import ProjectFormFields from './components/ProjectFormFields';
import { useNavigate, useLocation } from 'react-router-dom';
import { calculateTruss } from '../../services/api';
import { Button, Box } from '@mui/material';
import { saveAs } from 'file-saver';
import CircularProgress from '@mui/material/CircularProgress';

const NewProject = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const importedValues = location.state?.importedValues;
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (importedValues) {
      setLoading(true);
      setError('');
      calculateTruss(importedValues)
        .then((results) => {
          navigate('/results', { state: { results, params: importedValues } });
        })
        .catch((error) => {
          setError(error.response?.data?.message || error.message || 'Erreur inconnue');
          setLoading(false);
        });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [importedValues]);

  const handleSubmit = async (values) => {
    setLoading(true);
    setError('');
    try {
      const results = await calculateTruss(values);
      navigate('/results', { state: { results, params: values } });
    } catch (error) {
      setError(error.response?.data?.message || error.message || 'Erreur inconnue');
    } finally {
      setLoading(false);
    }
  };

  // Fonction d'export JSON
  const handleExport = (values) => {
    const blob = new Blob([JSON.stringify(values, null, 2)], { type: 'application/json' });
    saveAs(blob, `${values.project_name || 'projet_charpente'}.json`);
  };

  if (importedValues && loading) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 8 }}>
        <CircularProgress />
        <Box sx={{ mt: 2 }}>Chargement du projet...</Box>
      </Box>
    );
  }

  return (
    <Box>
      <Button 
        variant="outlined" 
        color="primary" 
        onClick={() => navigate('/')} 
        sx={{ mb: 2, width: { xs: '100%', sm: 'auto' } }}
      >
        Retour à l'accueil
      </Button>
      <ProjectFormFields 
        onSubmit={handleSubmit} 
        loading={loading} 
        externalError={error} 
        onExport={handleExport}
        initialState={importedValues}
      />
    </Box>
  );
};

export default NewProject; 