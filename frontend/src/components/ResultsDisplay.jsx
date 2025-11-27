import React from 'react';

const ResultsDisplay = ({ results, mode }) => {
  const modelOrder = mode === 'text'
    ? ['CNN-1D', 'BiLSTM', 'SVM', 'XGBoost', 'Naive Bayes', 'KNN', 'Random Forest']
    : ['Deep CNN', 'MLP', 'SVM', 'XGBoost', 'Random Forest', 'KNN', 'Naive Bayes'];

  const getEmotionColor = (emotion) => {
    const colors = {
      'happy': 'bg-green-500',
      'joy': 'bg-green-500',
      'love': 'bg-pink-500',
      'surprise': 'bg-yellow-500',
      'neutral': 'bg-gray-500',
      'sad': 'bg-blue-500',
      'sadness': 'bg-blue-500',
      'anger': 'bg-red-500',
      'angry': 'bg-red-500',
      'fear': 'bg-purple-500',
      'disgust': 'bg-orange-500'
    };

    return colors[emotion.toLowerCase()] || 'bg-gray-500';
  };

  return (
    <div className="space-y-4">
      <h3 className="text-xl font-bold text-gray-800 border-b-2 border-gray-300 pb-2">
        Prediction Results
      </h3>

      <div className="overflow-x-auto">
        <table className="min-w-full bg-white border border-gray-300">
          <thead className="bg-gray-100">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider border-b">
                Model
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider border-b">
                Predicted Emotion
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider border-b">
                Confidence
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {modelOrder.map((model) => {
              const result = results.predictions[model];

              if (!result) return null;

              return (
                <tr key={model} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {model}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full text-white ${getEmotionColor(result.emotion)}`}>
                      {result.emotion.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                    <div className="flex items-center">
                      <div className="w-24 bg-gray-200 rounded-full h-2 mr-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full"
                          style={{ width: `${result.confidence}%` }}
                        ></div>
                      </div>
                      <span className="font-semibold">{result.confidence.toFixed(1)}%</span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="mt-4 p-4 bg-blue-50 rounded-lg">
        <h4 className="font-semibold text-gray-800 mb-2">Summary:</h4>
        <p className="text-sm text-gray-700">
          {mode === 'text' ? 'Text' : 'Facial'} analysis complete!
          {Object.keys(results.predictions).length} models analyzed.
        </p>
      </div>
    </div>
  );
};

export default ResultsDisplay;