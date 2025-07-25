import React, { useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';

const WelcomeScreen = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef();

  // Handler pour l'import JSON
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const data = JSON.parse(event.target.result);
        navigate('/new-project', { state: { importedValues: data } });
      } catch (err) {
        alert('Fichier JSON invalide.');
      }
    };
    reader.readAsText(file);
  };

  return (
    <Box sx={{ textAlign: 'center', mt: { xs: 4, sm: 8 }, px: { xs: 2, sm: 0 } }}>
      <Typography 
        variant="h3" 
        fontWeight={700} 
        gutterBottom 
        sx={{ fontSize: { xs: '2rem', sm: '3rem' } }}
      >
        Bienvenue sur TrussAnalyzer
      </Typography>
      <Typography 
        variant="subtitle1" 
        color="text.secondary" 
        gutterBottom
        sx={{ fontSize: { xs: '1rem', sm: '1.25rem' } }}
      >
        Analysez, visualisez et exportez vos charpentes en toute simplicité.
      </Typography>
      <Stack 
        direction={{ xs: 'column', sm: 'row' }} 
        spacing={{ xs: 2, sm: 3 }} 
        justifyContent="center" 
        mt={{ xs: 3, sm: 5 }}
      >
        <Button
          variant="contained"
          color="primary"
          size="large"
          startIcon={<AddCircleOutlineIcon />}
          onClick={() => navigate('/new-project')}
          sx={{ minWidth: 200 }}
        >
          Nouveau projet
        </Button>
        <Button
          variant="outlined"
          color="primary"
          size="large"
          startIcon={<FolderOpenIcon />}
          onClick={() => fileInputRef.current && fileInputRef.current.click()}
          sx={{ minWidth: 200 }}
        >
          Ouvrir projet
        </Button>
        <input
          type="file"
          accept="application/json"
          ref={fileInputRef}
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />
      </Stack>
    </Box>
  );
};

export default WelcomeScreen; 