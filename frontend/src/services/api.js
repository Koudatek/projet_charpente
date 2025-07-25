import axios from 'axios';

const BASE_URL = 'http://localhost:5000'; // À adapter pour la prod

export const calculateTruss = async (params) => {
  const response = await axios.post(`${BASE_URL}/calculate`, params);
  return response.data;
};

export const generatePdf = async (params) => {
  // Pour récupérer un PDF en blob
  const response = await axios.post(`${BASE_URL}/generate_pdf`, params, {
    responseType: 'blob',
  });
  return response.data;
}; 