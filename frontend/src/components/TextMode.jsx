import React, { useState } from 'react';
import axios from 'axios';
import ResultsDisplay from './ResultsDisplay';

const TextMode = () => {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!text.trim()) {
      setError('Please enter some text');
      return;
    }

    setLoading(true);
    setError('');
    setResults(null);

    try {
      const response = await axios.post('http://localhost:8000/predict/text', {
        text: text
      });

      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error predicting emotion. Make sure backend is running!');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Text Emotion Analysis</h2>

        <form onSubmit={handleSubmit}>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to analyze emotions... (e.g., 'I am so happy today!')"
            className="w-full h-32 px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none resize-none"
            disabled={loading}
          />

          <button
            type="submit"
            disabled={loading}
            className="mt-4 w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-semibold py-3 px-6 rounded-lg transition duration-200"
          >
            {loading ? 'Analyzing...' : 'Analyze Emotion'}
          </button>
        </form>

        {error && (
          <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
            {error}
          </div>
        )}

        {results && (
          <div className="mt-6">
            <ResultsDisplay results={results} mode="text" />
          </div>
        )}
      </div>
    </div>
  );
};

export default TextMode;