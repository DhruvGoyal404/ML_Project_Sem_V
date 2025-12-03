import React, { useState, useRef } from 'react';
import axios from 'axios';
import ResultsDisplay from './ResultsDisplay';

const ImageMode = () => {
  const [_image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');

  const fileInputRef = useRef(null);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];

    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      predictEmotion(file);
    }
  };

  const predictEmotion = async (imageBlob) => {
    setLoading(true);
    setError('');
    setResults(null);

    try {
      const formData = new FormData();
      formData.append('file', imageBlob);

      const response = await axios.post('http://localhost:8000/predict/image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error predicting emotion. Make sure backend is running!');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setImage(null);
    setPreview(null);
    setResults(null);
    setError('');
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Facial Emotion Analysis</h2>

        {!preview && (
          <div className="flex justify-center">
            <button
              onClick={() => fileInputRef.current.click()}
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-8 px-12 rounded-lg transition duration-200"
            >
              <svg className="w-12 h-12 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              Upload Image
            </button>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileUpload}
          className="hidden"
        />

        {preview && (
          <div className="mt-4">
            <img src={preview} alt="Captured" className="w-full max-w-md mx-auto rounded-lg shadow-lg" />
            <button
              onClick={reset}
              className="mt-4 w-full bg-gray-600 hover:bg-gray-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
            >
              Try Another Image
            </button>
          </div>
        )}

        {loading && (
          <div className="mt-4 text-center">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            <p className="mt-2 text-gray-600">Analyzing emotion...</p>
          </div>
        )}

        {error && (
          <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
            {error}
          </div>
        )}

        {results && (
          <div className="mt-6">
            <ResultsDisplay results={results} mode="image" />
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageMode;