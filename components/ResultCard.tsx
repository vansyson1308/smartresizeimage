import React from 'react';
import { ProcessedImage, AVAILABLE_CONFIGS } from '../types';
import { downloadImage } from '../utils/imageUtils';

interface ResultCardProps {
  item: ProcessedImage;
}

export const ResultCard: React.FC<ResultCardProps> = ({ item }) => {
  const config = AVAILABLE_CONFIGS.find(c => c.id === item.configId);

  const handleDownload = () => {
    if (item.resultUrl) {
      downloadImage(item.resultUrl, `autobanner-${item.configId}-${Date.now()}.png`);
    }
  };

  return (
    <div className="bg-slate-800 rounded-xl overflow-hidden shadow-lg border border-slate-700 flex flex-col h-full">
      <div className="p-4 border-b border-slate-700 flex justify-between items-center bg-slate-900/50">
        <h3 className="font-semibold text-white text-sm">{config?.label}</h3>
        <span className={`text-xs px-2 py-1 rounded-full ${
          item.status === 'completed' ? 'bg-green-900/30 text-green-400 border border-green-800' :
          item.status === 'processing' ? 'bg-indigo-900/30 text-indigo-400 border border-indigo-800 animate-pulse' :
          item.status === 'failed' ? 'bg-red-900/30 text-red-400 border border-red-800' :
          'bg-slate-700 text-slate-400'
        }`}>
          {item.status.toUpperCase()}
        </span>
      </div>
      
      <div className="flex-1 bg-slate-950 relative min-h-[300px] flex items-center justify-center p-4">
        {item.status === 'completed' && item.resultUrl ? (
          <img 
            src={item.resultUrl} 
            alt="Result" 
            className="max-w-full max-h-[400px] object-contain rounded-md shadow-2xl"
          />
        ) : item.status === 'processing' ? (
          <div className="text-center">
            <div className="w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-slate-400 text-sm animate-pulse">Generating layout...</p>
            <p className="text-slate-500 text-xs mt-2">AI is adjusting mascot & text</p>
          </div>
        ) : item.status === 'failed' ? (
          <div className="text-center px-4">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-red-500 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-red-400 text-sm">Failed to generate</p>
            <p className="text-slate-500 text-xs mt-1">{item.error || "Unknown error"}</p>
          </div>
        ) : (
          <p className="text-slate-600 text-sm">Waiting to start...</p>
        )}
      </div>

      {item.status === 'completed' && (
        <div className="p-4 border-t border-slate-700 bg-slate-900/50">
          <button 
            onClick={handleDownload}
            className="w-full flex items-center justify-center space-x-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm py-2 rounded-lg transition-colors"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            <span>Download</span>
          </button>
        </div>
      )}
    </div>
  );
};