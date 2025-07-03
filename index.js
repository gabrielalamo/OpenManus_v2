const express = require('express');
const dotenv = require('dotenv');
const axios = require('axios');

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.static('public'));

// Basic route
app.get('/', (req, res) => {
  res.send('OpenManus API is running');
});

// Simple agent endpoint
app.post('/api/agent', async (req, res) => {
  try {
    const { prompt } = req.body;
    
    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required' });
    }
    
    // This is a placeholder for actual OpenManus functionality
    // In a real implementation, this would connect to your LLM and tools
    const response = {
      result: `Processed: ${prompt}`,
      timestamp: new Date().toISOString()
    };
    
    res.json(response);
  } catch (error) {
    console.error('Error processing agent request:', error);
    res.status(500).json({ error: 'Failed to process request' });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`OpenManus server running on port ${PORT}`);
});