import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export const calculateTruss = async (params) => {
  const response = await axios.post(`${API_URL}/calculate`, params);
  return response.data;
};

export const generatePdf = async (params) => {
  // Pour récupérer un PDF en blob
  const response = await axios.post(`${API_URL}/generate_pdf`, params, {
    responseType: 'blob',
  });
  return response.data;
}; 