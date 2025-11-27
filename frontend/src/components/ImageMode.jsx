import React, { useState, useRef } from 'react';
import axios from 'axios';
import ResultsDisplay from './ResultsDisplay';

const ImageMode = () => {
  const [mode, setMode] = useState(null); // 'upload' or 'camera'
  const [_image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [stream, setStream] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }

      setStream(mediaStream);
      setMode('camera');
      setError('');
    } catch (err) {
      setError('Cannot access camera. Please check permissions.');
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setMode(null);
  };

  const captureImage = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video && canvas) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);

      canvas.toBlob(async (blob) => {
        setImage(blob);
        setPreview(canvas.toDataURL());
        await predictEmotion(blob);
      }, 'image/jpeg');
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];

    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setMode('upload');
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
    stopCamera();
    setImage(null);
    setPreview(null);
    setResults(null);
    setError('');
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Facial Emotion Analysis</h2>

        {!mode && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <button
              onClick={() => fileInputRef.current.click()}
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-8 px-6 rounded-lg transition duration-200"
            >
              <svg className="w-12 h-12 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              Upload Image
            </button>

            <button
              onClick={startCamera}
              className="bg-green-600 hover:bg-green-700 text-white font-semibold py-8 px-6 rounded-lg transition duration-200"
            >
              <svg className="w-12 h-12 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              Use Camera
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

        {mode === 'camera' && !preview && (
          <div className="relative">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-full rounded-lg"
            />
            <div className="mt-4 flex gap-4">
              <button
                onClick={captureImage}
                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition duration-200"
              >
                Capture & Analyze
              </button>
              <button
                onClick={stopCamera}
                className="bg-red-600 hover:bg-red-700 text-white font-semibold py-3 px-6 rounded-lg transition duration-200"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        <canvas ref={canvasRef} className="hidden" />

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