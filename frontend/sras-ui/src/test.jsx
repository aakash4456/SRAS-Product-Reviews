// import { useState } from 'react';
// import axios from 'axios';

// export default function App() {
//   const [review, setReview] = useState('');
//   const [rating, setRating] = useState(3);
//   const [result, setResult] = useState(null);
//   const [error, setError] = useState('');

//   const handleSubmit = async () => {
//     try {
//       const res = await axios.post('http://localhost:8000/analyze', {
//         review,
//         rating: parseInt(rating)
//       });
//       setResult(res.data);
//       setError('');
//     } catch (err) {
//       setError('Something went wrong');
//       setResult(null);
//     }
//   };

//   return (
//     <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-6">
//       <div className="bg-white p-6 rounded-2xl shadow-md w-full max-w-md">
//         <h1 className="text-2xl font-bold mb-4">SRAS - Review Analyzer</h1>
//         <textarea
//           className="w-full border p-2 rounded mb-3"
//           rows="5"
//           placeholder="Enter your review..."
//           value={review}
//           onChange={(e) => setReview(e.target.value)}
//         ></textarea>
//         <label className="block mb-2">Rating (1-5):</label>
//         <input
//           type="number"
//           min="1"
//           max="5"
//           className="w-full border p-2 rounded mb-4"
//           value={rating}
//           onChange={(e) => setRating(e.target.value)}
//         />
//         <button
//           onClick={handleSubmit}
//           className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 rounded"
//         >
//           Analyze
//         </button>

//         {error && <p className="text-red-500 mt-4">{error}</p>}
//         {result && (
//           <div className="mt-4 p-4 border rounded bg-gray-50">
//             <p><strong>Sentiment:</strong> {result.sentiment}</p>
//             <p><strong>Relevance:</strong> {result.relevance}</p>
//           </div>
//         )}
//       </div>
//     </div>
//   );
// }