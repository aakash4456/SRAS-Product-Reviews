import { useState } from 'react';
import axios from 'axios';

export default function App() {
  const [review, setReview] = useState('');
  const [rating, setRating] = useState(3);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleSubmit = async () => {
    try {
      setError('');
      setResult(null);
      const res = await axios.post('http://localhost:8000/analyze', {
        review,
        rating: parseInt(rating),
      });
      setResult(res.data);
    } catch (err) {
      setError('Something went wrong');
      setResult(null);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#060f3a] px-4 py-8">
      <div className="bg-white rounded-3xl shadow-2xl p-8 w-full max-w-xl animate-fade-in">
        <h1 className="text-3xl font-bold text-center text-[#009B4D] mb-2 tracking-wide">
          SRAS
        </h1>
        <p className="text-center text-sm text-red-700 font-semibold mb-6 italic">
          Sentiment and Rating Alignment System
        </p>

        <textarea
          className="w-full border border-[#009B4D] focus:ring-2 focus:ring-[#FFCC00] p-4 rounded-lg mb-4 placeholder-gray-500 transition duration-200"
          rows="5"
          placeholder="Enter your product review..."
          value={review}
          onChange={(e) => setReview(e.target.value)}
        ></textarea>

        <label className="block mb-1 text-sm font-medium text-gray-700">
          Rating (1-5):
        </label>
        <input
          type="number"
          min="1"
          max="5"
          className="w-full border border-[#009B4D] focus:ring-2 focus:ring-[#FFCC00] p-3 rounded-lg mb-4 text-gray-800"
          value={rating}
          onChange={(e) => setRating(e.target.value)}
        />

        <button
          onClick={handleSubmit}
          className="w-full bg-[#009B4D] hover:bg-[#007A3C] text-white font-semibold py-3 px-4 rounded-lg shadow-md transition duration-300 transform hover:scale-105"
        >
          🔍 Analyze Review
        </button>

        {error && (
          <p className="text-red-600 text-center mt-4 bg-red-100 p-3 rounded">
            {error}
          </p>
        )}

        {result && (
          <div className="mt-6 p-5 border-2 border-[#FFCC00] rounded-xl bg-[#FFFBE5] shadow-inner animate-slide-up">
            <h2 className="text-xl font-bold text-[#009B4D] mb-3">
              🎯 Analysis Result
            </h2>
            <p className="mb-2">
              <strong className="text-gray-700">Sentiment:</strong>{' '}
              <span
                className={`font-semibold ${
                  result.sentiment === 'Positive'
                    ? 'text-green-600'
                    : 'text-yellow-700'
                }`}
              >
                {result.sentiment}
              </span>
            </p>
            <p>
              <strong className="text-gray-700">Relevance:</strong>{' '}
              <span
                className={`font-semibold ${
                  result.relevance === 'Aligned'
                    ? 'text-blue-600'
                    : 'text-orange-600'
                }`}
              >
                {result.relevance}
              </span>
            </p>
          </div>
        )}
      </div>

      {/* Animation Styles */}
      <style jsx>{`
        .animate-fade-in {
          animation: fadeIn 0.8s ease-in-out;
        }

        .animate-slide-up {
          animation: slideUp 0.7s ease-out;
        }

        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: scale(0.96);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }

        @keyframes slideUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
}








