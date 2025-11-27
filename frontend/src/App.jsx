import React, { useState } from 'react';
import Header from './components/Header';
import TextMode from './components/TextMode';
import ImageMode from './components/ImageMode';

function App() {
  const [activeMode, setActiveMode] = useState(null); // 'text' or 'image'

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-100 to-gray-200">
      <Header />

      <main className="container mx-auto px-4 py-8">
        {/* Mode Selection */}
        {!activeMode && (
          <div className="max-w-2xl mx-auto">
            <h2 className="text-3xl font-bold text-center text-gray-800 mb-8">
              Choose Analysis Mode
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <button
                onClick={() => setActiveMode('text')}
                className="bg-white hover:shadow-2xl shadow-lg rounded-lg p-8 transition duration-300 transform hover:scale-105"
              >
                <div className="text-blue-600 mb-4">
                  <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-gray-800 mb-2">Text Analysis</h3>
                <p className="text-gray-600">Analyze emotions from text using 7 trained models</p>
              </button>

              <button
                onClick={() => setActiveMode('image')}
                className="bg-white hover:shadow-2xl shadow-lg rounded-lg p-8 transition duration-300 transform hover:scale-105"
              >
                <div className="text-green-600 mb-4">
                  <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-gray-800 mb-2">Facial Analysis</h3>
                <p className="text-gray-600">Detect emotions from facial expressions using 7 models</p>
              </button>
            </div>
          </div>
        )}

        {/* Active Mode */}
        {activeMode && (
          <div>
            <button
              onClick={() => setActiveMode(null)}
              className="mb-6 flex items-center text-blue-600 hover:text-blue-800 font-semibold"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Back to Mode Selection
            </button>

            {activeMode === 'text' && <TextMode />}
            {activeMode === 'image' && <ImageMode />}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-6 mt-12">
        <div className="container mx-auto px-4 text-center">
          <p className="text-sm">
            © 2024 Thapar Institute of Engineering & Technology
          </p>
          <p className="text-xs text-gray-400 mt-2">
            MultiModal Emotion Detection System | Machine Learning Project
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;