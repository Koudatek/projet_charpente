import React, { useState } from 'react';
import { TextField, Button, MenuItem, Box, Typography, Alert } from '@mui/material';
import Grid from '@mui/material/Grid';

// Options for the geometry type dropdown
const geometryOptions = [
  { value: 'howe_brise', label: 'Howe brisé' },
  { value: 'pratt_brise', label: 'Pratt brisé' },
  { value: 'warren_brise_poincon', label: 'Warren brisé avec poinçon' },
];

// Initial state for form fields
const initialState = {
  project_name: 'Projet Test',
  engineer_name: 'John Doe',
  geometry_type: 'howe_brise',
  L: '5',
  H: '2',
  m: '4',
  alpha_deg: '30',
  F: '1000',
};

const ProjectFormFields = ({ onSubmit, loading = false, externalError = '', onExport, initialState: importedInitialState }) => {
  const [values, setValues] = useState(importedInitialState || initialState);
  const [error, setError] = useState('');

  // Handle input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setValues((prev) => ({ ...prev, [name]: value }));
  };

  // Validate form inputs
  const validate = () => {
    if (!values.project_name || !values.engineer_name) return 'Nom du projet et ingénieur requis';
    if (!['howe_brise', 'pratt_brise', 'warren_brise_poincon'].includes(values.geometry_type)) return 'Géométrie invalide';
    if (isNaN(Number(values.L)) || Number(values.L) <= 0) return 'L doit être un nombre positif';
    if (isNaN(Number(values.H)) || Number(values.H) <= 0) return 'H doit être un nombre positif';
    if (isNaN(Number(values.m)) || Number(values.m) <= 0 || Number(values.m) % 2 !== 0) return 'm doit être un nombre pair positif';
    if (isNaN(Number(values.alpha_deg))) return 'Angle invalide';
    if (isNaN(Number(values.F)) || Number(values.F) <= 0) return 'F doit être un nombre positif';
    return '';
  };

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    const err = validate();
    if (err) {
      setError(err);
      return;
    }
    setError('');
    if (onSubmit) onSubmit(values);
  };

  return (
    <Box component="form" onSubmit={handleSubmit} sx={{ mt: 2, px: { xs: 1, sm: 0 } }}>
      <Typography 
        variant="h5" 
        fontWeight={600} 
        mb={2}
        sx={{ fontSize: { xs: '1.3rem', sm: '2rem' } }}
      >
        Créer un nouveau projet
      </Typography>
      {(error || externalError) && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error || externalError}
        </Alert>
      )}
      <Grid container columns={12} spacing={2}>
        <Grid gridColumn={{ xs: 'span 12', sm: 'span 6' }}>
          <TextField
            label="Nom du projet"
            name="project_name"
            value={values.project_name}
            onChange={handleChange}
            fullWidth
            required
            disabled={loading}
          />
        </Grid>
        <Grid gridColumn={{ xs: 'span 12', sm: 'span 6' }}>
          <TextField
            label="Nom de l'ingénieur"
            name="engineer_name"
            value={values.engineer_name}
            onChange={handleChange}
            fullWidth
            required
            disabled={loading}
          />
        </Grid>
        <Grid gridColumn={{ xs: 'span 12', sm: 'span 6' }}>
          <TextField
            select
            label="Type de géométrie"
            name="geometry_type"
            value={values.geometry_type}
            onChange={handleChange}
            fullWidth
            disabled={loading}
          >
            {geometryOptions.map((option) => (
              <MenuItem key={option.value} value={option.value}>
                {option.label}
              </MenuItem>
            ))}
          </TextField>
        </Grid>
        <Grid gridColumn={{ xs: 'span 12', sm: 'span 6' }}>
          <TextField
            label="Demi-portée L (m)"
            name="L"
            value={values.L}
            onChange={handleChange}
            fullWidth
            required
            type="number"
            inputProps={{ min: 0 }}
            disabled={loading}
          />
        </Grid>
        <Grid gridColumn={{ xs: 'span 12', sm: 'span 6' }}>
          <TextField
            label="Hauteur H (m)"
            name="H"
            value={values.H}
            onChange={handleChange}
            fullWidth
            required
            type="number"
            inputProps={{ min: 0 }}
            disabled={loading}
          />
        </Grid>
        <Grid gridColumn={{ xs: 'span 12', sm: 'span 6' }}>
          <TextField
            label="Nombre de divisions m (pair)"
            name="m"
            value={values.m}
            onChange={handleChange}
            fullWidth
            required
            type="number"
            inputProps={{ min: 2, step: 2 }}
            disabled={loading}
          />
        </Grid>
        <Grid gridColumn={{ xs: 'span 12', sm: 'span 6' }}>
          <TextField
            label="Angle alpha/beta (°)"
            name="alpha_deg"
            value={values.alpha_deg}
            onChange={handleChange}
            fullWidth
            required
            type="number"
            disabled={loading}
          />
        </Grid>
        <Grid gridColumn={{ xs: 'span 12', sm: 'span 6' }}>
          <TextField
            label="Charge nodale F (N)"
            name="F"
            value={values.F}
            onChange={handleChange}
            fullWidth
            required
            type="number"
            inputProps={{ min: 0 }}
            disabled={loading}
          />
        </Grid>
      </Grid>
      <Button 
        type="submit" 
        variant="contained" 
        color="primary" 
        size="large" 
        sx={{ mt: { xs: 2, sm: 3 }, width: { xs: '100%', sm: 'auto' } }} 
        disabled={loading}
      >
        {loading ? 'Calcul en cours...' : 'Lancer le calcul'}
      </Button>
    </Box>
  );
};

export default ProjectFormFields;